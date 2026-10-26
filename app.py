import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import streamlit as st

from config import GEMINI_API_KEY, GEMINI_MODEL
import src.ai_service as ai_service
from src.ai_service import analyze_issue, generate_complaint, run_contradiction_audit
from src.schemas import ContradictionAudit, StructuredIssue
from src.scoring import calculate_quality_score, count_unsupported_sentences
from src.speech_service import transcribe_audio
from src.utils import load_image_pil, save_temp_audio

logger = logging.getLogger(__name__)

st.set_page_config(page_title="CivicFix", page_icon="🏛️", layout="wide")

SEVERITY_COLORS = {
    # background, text
    "Low": ("#2e7d32", "#ffffff"),
    "Medium": ("#fbc02d", "#1a1a1a"),
    "High": ("#ef6c00", "#ffffff"),
    "Critical": ("#c62828", "#ffffff"),
}

EVAL_REPORT_PATH = Path("eval/results/eval_report.json")
DEMO_CACHE_DIR = Path("demo_cache")  # Gemini outputs per input, so demo rehearsals don't burn the free-tier quota

ARCHITECTURE = """\
Voice → Whisper → Transcript
Image → Gemini → Evidence
Text + Transcript + Evidence
      → Gemini → Structured Issue
Structured Issue → DistilBERT
      → Category + Severity
User Claims + Evidence
      → MiniLM NLI → Contradiction Audit
Verified Facts → Gemini → Complaint
Complaint → Scoring → Quality Score"""


class AnalysisError(Exception):
    """Raised with a user-facing message; shown in the UI instead of a stack trace."""


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "permission_denied" in low or "unauthenticated" in low:
        return "Gemini API key is missing or invalid. Check `GEMINI_API_KEY` in your `.env` file, then restart the app."
    if "resource_exhausted" in low or "429" in low or "quota" in low:
        return ("Gemini free-tier quota reached on all configured models (20 requests/day per model). "
                "Try again later, or add more models to `GEMINI_FALLBACK_MODELS` in `.env`.")
    if ("not_found" in low or "404" in low) and "model" in low:
        return f"Gemini model `{GEMINI_MODEL}` is not available. Update `GEMINI_MODEL` in your `.env` file."
    if "unavailable" in low or "503" in low or "overloaded" in low:
        return "Gemini is temporarily unavailable. Please try again in a moment."
    return f"Analysis failed: {msg[:300]}"


def severity_badge(severity: str) -> str:
    bg, fg = SEVERITY_COLORS.get(severity, ("#757575", "#ffffff"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 12px;border-radius:12px;'
        f'font-weight:600;font-size:0.9rem;">{severity}</span>'
    )


def bullet_list(items):
    if items:
        st.markdown("\n".join(f"- {i}" for i in items))
    else:
        st.caption("None")


def render_audit(audit):
    n_supported = sum(f.status == "supported" for f in audit.flags)
    m1, m2, m3 = st.columns(3)
    m1.metric("✅ Verified", n_supported)
    m2.metric("❌ Contradicted", len(audit.contradicted_claims))
    m3.metric("❔ Unverifiable", len(audit.unverifiable_claims))
    if audit.flags:
        icons = {"supported": "✅", "contradicted": "❌", "unverifiable": "❔"}
        with st.expander("Claim-by-claim details"):
            for f in audit.flags:
                st.markdown(f"{icons.get(f.status, '•')} **{f.claim}**  \n_{f.reason}_")
    else:
        st.caption("No user claims to check.")


def run_custom_models(issue, text, transcript) -> dict:
    """DistilBERT category/severity + MiniLM NLI audit. Never raises — custom models are an add-on to Gemini."""
    out = {"custom_category": None, "custom_severity": None, "custom_input": None, "nli_audit": None, "custom_error": None}
    # DistilBERT was trained on raw complaint text; for image-only input fall back to Gemini's problem summary
    out["custom_input"] = " ".join(t for t in (text, transcript) if t) or issue.problem
    try:
        # Lazy imports: torch/transformers load only when an analysis actually runs
        from models.classifier import predict as custom_predict
        out["custom_category"] = custom_predict(out["custom_input"], "category")
        out["custom_severity"] = custom_predict(out["custom_input"], "severity")
    except Exception as e:
        logger.exception("DistilBERT prediction failed")
        out["custom_error"] = f"DistilBERT: {e}"
    try:
        from src.contradiction import run_nli_audit
        out["nli_audit"] = run_nli_audit(issue)
    except Exception as e:
        logger.exception("MiniLM NLI audit failed")
        out["custom_error"] = ((out["custom_error"] + " · ") if out["custom_error"] else "") + f"MiniLM: {e}"
    return out


def build_text_report(analysis: dict, complaint: str) -> str:
    issue, quality = analysis["issue"], analysis["quality"]
    lines = [
        "CIVICFIX REPORT",
        "===============",
        f"Problem: {issue.problem}",
        f"Category: {issue.category}" + (f" / {issue.subcategory}" if issue.subcategory else ""),
        f"Severity: {issue.severity.value}",
        f"Department: {issue.department}",
        f"Location: {issue.location or 'Not specified'}",
        f"Duration: {issue.duration or 'Not specified'}",
        f"Recommended action: {issue.recommended_action}",
        "",
        "COMPLAINT:",
        complaint,
        "",
    ]
    if quality is not None:
        lines.append(f"QUALITY SCORE: {quality.overall:.0f}/100")
    lines.append(f"GENERATED: {date.today().isoformat()}")
    return "\n".join(lines) + "\n"


def load_eval_report():
    try:
        return json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("Could not read eval report")
        return None


def _accuracy(block: dict) -> float:
    # classification_report includes "accuracy" when every label is present; weighted recall equals it otherwise
    return block["per_class"].get("accuracy", block["weighted_recall"])


def render_eval_summary():
    report = load_eval_report()
    if report is None:
        st.info("Run eval suite first: `python -m eval.eval_runner`")
        return
    cls, sev, con = report["classification_accuracy"], report["severity_accuracy"], report["contradiction_detection"]
    st.markdown("**Model Comparison**")
    st.table({
        "Metric": ["Category Accuracy", "Severity Accuracy", "Contradiction P/R"],
        "Gemini": [f"{_accuracy(cls['gemini']):.0%}", f"{_accuracy(sev['gemini']):.0%}",
                   f"{con['gemini']['contradiction_precision']:.1f} / {con['gemini']['contradiction_recall']:.1f}"],
        "Custom": [f"{_accuracy(cls['custom_distilbert']):.0%}", f"{_accuracy(sev['custom_distilbert']):.0%}",
                   f"{con['custom_minilm']['contradiction_precision']:.1f} / "
                   f"{con['custom_minilm']['contradiction_recall']:.1f}"],
    })
    hal = report["hallucination_reduction"]
    st.markdown("**Hallucination Reduction**")
    st.markdown(f"With audit: **{hal['avg_unsupported_rate_with_audit']:.1%}** unsupported  \n"
                f"Without audit: **{hal['avg_unsupported_rate_without_audit']:.1%}** unsupported")
    st.caption(f"{report.get('evaluated_cases', report['total_cases'])} test cases · custom = DistilBERT (category, "
               f"severity) and MiniLM NLI (contradiction) · avg quality score {report['avg_quality_score']:.1f}")


def _cache_path(text, transcript, image_file) -> Path:
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8") + b"\0" + (transcript or "").encode("utf-8") + b"\0")
    if image_file is not None:
        image_file.seek(0)
        h.update(image_file.read())
        image_file.seek(0)
    return DEMO_CACHE_DIR / f"{h.hexdigest()[:16]}.json"


def load_cached_gemini(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (StructuredIssue.model_validate(data["issue"]), ContradictionAudit.model_validate(data["audit"]),
                data["complaint"], data["model_used"])
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("Ignoring unreadable demo cache entry %s", path)
        return None


def save_cached_gemini(path: Path, issue, audit, complaint, model_used):
    try:
        DEMO_CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps({"issue": issue.model_dump(mode="json"), "audit": audit.model_dump(mode="json"),
                                    "complaint": complaint, "model_used": model_used}, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Could not write demo cache entry")


def run_pipeline(text_input, audio_file, image_file, use_cache=False) -> dict:
    text = text_input.strip()
    result = {
        "has_text": bool(text),
        "has_audio": audio_file is not None,
        "has_image": image_file is not None,
        "transcription": None,
        "issue": None,
        "audit": None,
        "complaint": None,
        "quality": None,
        "unsupported": None,  # (unsupported_count, total_count)
        "custom": None,       # run_custom_models() output
        "model_used": None,   # Gemini model that served the last request (fallback may switch)
        "original_input": None,  # what the user actually said, for the before/after panel
        "error": None,
    }

    with st.status("Analyzing issue...", expanded=True) as status:
        try:
            status.write("✓ Input received")

            transcript = None
            if audio_file is not None:
                status.write("Transcribing voice recording...")
                audio_file.seek(0)
                audio_path = save_temp_audio(audio_file)
                try:
                    result["transcription"] = transcribe_audio(audio_path)
                finally:
                    audio_path.unlink(missing_ok=True)
                tr = result["transcription"]
                if tr["success"] and tr["transcript"]:
                    transcript = tr["transcript"]
                    status.write("✓ Speech transcribed")
                else:
                    status.write("⚠️ Speech transcription failed — continuing with text/image")

            result["original_input"] = "\n".join(t for t in (text, transcript) if t) or "(image only — no text or voice)"

            if not text and not transcript and image_file is None:
                raise AnalysisError("Nothing to analyze: voice transcription failed and no text or image was provided.")
            if not GEMINI_API_KEY:
                raise AnalysisError(
                    "Gemini API key not found. Copy `.env.example` to `.env`, set `GEMINI_API_KEY`, then restart the app."
                )

            cache_path = _cache_path(text, transcript, image_file)
            cached = load_cached_gemini(cache_path) if use_cache else None

            if cached is not None:
                issue, audit, result["complaint"], cached_model = cached
                result["issue"], result["audit"] = issue, audit
                result["from_cache"] = True
                status.write("✓ Gemini results loaded from demo cache (identical input analyzed before)")
                result["custom"] = run_custom_models(issue, text, transcript)
                status.write("✓ Custom models (DistilBERT + MiniLM NLI) run")
            else:
                image_pil = None
                if image_file is not None:
                    image_file.seek(0)
                    image_pil = load_image_pil(image_file)

                issue = analyze_issue(text or None, transcript, image_pil)
                result["issue"] = issue
                if image_pil is not None:
                    status.write("✓ Image analyzed")
                status.write("✓ Issue classified")

                result["custom"] = run_custom_models(issue, text, transcript)
                status.write("✓ Custom models (DistilBERT + MiniLM NLI) run")

                audit = run_contradiction_audit(issue)
                result["audit"] = audit
                status.write("✓ Contradiction audit complete")

                result["complaint"] = generate_complaint(issue, audit)
                status.write("✓ Complaint generated")
                cached_model = ai_service.last_model_used
                save_cached_gemini(cache_path, issue, audit, result["complaint"], cached_model)

            result["quality"] = calculate_quality_score(
                issue, audit, text, image_file is not None, audio_file is not None
            )
            result["unsupported"] = count_unsupported_sentences(result["complaint"], audit.verified_facts)
            status.write("✓ Quality scored")
            result["model_used"] = cached_model

            status.update(label="Analysis complete", state="complete", expanded=False)
        except AnalysisError as e:
            result["error"] = str(e)
            status.update(label="Analysis stopped", state="error", expanded=False)
        except Exception as e:
            logger.exception("Analysis pipeline failed")
            result["error"] = friendly_error(e)
            status.update(label="Analysis failed", state="error", expanded=False)

    return result


# ---------- Session state ----------
def init_state():
    defaults = {
        "input_version": 0,   # bumped on reset so widgets get fresh keys (file_uploader can't be cleared directly)
        "analysis": None,     # populated by the AI pipeline in later phases
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all():
    version = st.session_state.get("input_version", 0)
    st.session_state.clear()
    st.session_state["input_version"] = version + 1


init_state()
v = st.session_state["input_version"]


# ---------- Sidebar ----------
with st.sidebar:
    use_demo_cache = st.toggle(
        "Reuse cached results for identical inputs", value=True,
        help="Saves Gemini quota (free tier: 20 requests/day per model). Turn off to force a fresh analysis.",
    )
    st.subheader("🧭 Architecture")
    st.code(ARCHITECTURE, language=None)
    st.subheader("📈 Evaluation Summary")
    render_eval_summary()


# ---------- Header ----------
st.title("🏛️ CivicFix")
st.markdown("**Turn everyday problems into actionable reports.**")
st.caption("AI-powered multimodal civic issue reporting")
st.divider()


# ---------- Layout ----------
input_col, results_col = st.columns([2, 3], gap="large")

with input_col:
    st.subheader("📝 Report an Issue")

    text_input = st.text_area(
        "Describe the problem",
        placeholder="e.g. The streetlight outside the hostel has not been working for 3 days, it is very dark at night...",
        help="Informal descriptions are fine — CivicFix turns them into a structured complaint.",
        height=150,
        key=f"text_input_{v}",
    )

    audio_file = st.file_uploader(
        "Upload Voice Recording",
        type=["wav", "mp3", "m4a", "ogg"],
        key=f"audio_file_{v}",
    )
    if audio_file is not None:
        st.audio(audio_file)

    image_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"image_file_{v}",
    )
    if image_file is not None:
        st.image(image_file, caption="Uploaded image", width="stretch")

    has_input = bool(text_input.strip()) or audio_file is not None or image_file is not None

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        analyze_clicked = st.button(
            "🔍 Analyze Issue",
            type="primary",
            disabled=not has_input,
            width="stretch",
        )
    with btn_col2:
        st.button("↺ Reset", on_click=reset_all, width="stretch")

    if not has_input:
        st.caption("Provide at least one input (text, voice, or image) to enable analysis.")

with results_col:
    st.subheader("📊 Analysis Results")

    if analyze_clicked:
        st.session_state["analysis"] = run_pipeline(text_input, audio_file, image_file, use_demo_cache)
        st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

    analysis = st.session_state.get("analysis")

    if analysis is None:
        with st.container(border=True):
            st.markdown("**🏷️ Category & Severity**")
            st.caption("Issue category and severity will appear here.")
        with st.container(border=True):
            st.markdown("**📄 Structured Complaint**")
            st.caption("A formal, actionable complaint will be generated here.")
        with st.container(border=True):
            st.markdown("**⚖️ Contradiction Check**")
            st.caption("Claims will be cross-checked against image evidence.")
        with st.container(border=True):
            st.markdown("**⭐ Quality Score**")
            st.caption("A deterministic report quality score will appear here.")
    else:
        transcription = analysis.get("transcription")
        if transcription is not None:
            if transcription["success"] and transcription["transcript"]:
                with st.expander("📝 Voice Transcript"):
                    st.write(transcription["transcript"])
                    if transcription["language"]:
                        st.caption(f"Detected language: {transcription['language']}")
            else:
                st.warning("Voice transcription failed — continuing with text/image.")

        if analysis["error"]:
            st.error(analysis["error"])

        issue = analysis["issue"]
        if issue is not None:
            # --- Category & Severity ---
            with st.container(border=True):
                st.markdown(f"#### {issue.problem}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption("Category")
                    st.markdown(f"**{issue.category}**" + (f"  \n{issue.subcategory}" if issue.subcategory else ""))
                with c2:
                    st.caption("Severity")
                    st.markdown(severity_badge(issue.severity.value), unsafe_allow_html=True)
                with c3:
                    st.caption("Department")
                    st.markdown(f"**{issue.department}**")
                st.markdown(f"**Severity reason:** {issue.severity_reason}")
                c4, c5, c6 = st.columns(3)
                c4.markdown(f"**📍 Location:** {issue.location or 'Not specified'}")
                c5.markdown(f"**⏱️ Duration:** {issue.duration or 'Not specified'}")
                c6.markdown(f"**🎯 Confidence:** {issue.confidence:.0%}")
                st.markdown(f"**Recommended action:** {issue.recommended_action}")

            # --- Before vs after ---
            quality = analysis["quality"]
            with st.container(border=True):
                st.markdown("**🔄 Before vs After**")
                b1, b2 = st.columns(2)
                with b1:
                    st.caption("Original input")
                    st.markdown(f"> {analysis.get('original_input') or '—'}")
                    if quality is not None:
                        st.markdown(f"**Raw Quality: {quality.raw_input_score:.0f}/100**")
                with b2:
                    st.caption("AI-structured report")
                    st.markdown(f"> {issue.problem} ({issue.category} / {issue.severity.value})")
                    if quality is not None:
                        st.markdown(f"**AI Quality: {quality.overall:.0f}/100**")

            # --- Gemini vs custom classification ---
            custom = analysis.get("custom") or {}
            with st.container(border=True):
                st.markdown("**🧪 Classification — Gemini vs Custom (DistilBERT)**")
                if custom.get("custom_category") is None and custom.get("custom_severity") is None:
                    st.caption(custom.get("custom_error") or
                               "Custom classifier not trained yet — run `python -m models.classifier --train`.")
                else:
                    def agree(gemini_value, custom_value):
                        # Gemini sometimes appends a subcategory ("Electrical / Lighting"), so prefix-match
                        return "✅" if custom_value and gemini_value.lower().startswith(custom_value.lower()) else "≠"

                    h0, h1, h2, h3 = st.columns([1.1, 2, 2, 0.6])
                    h1.caption("Gemini")
                    h2.caption("Custom (DistilBERT)")
                    r0, r1, r2, r3 = st.columns([1.1, 2, 2, 0.6])
                    r0.markdown("**Category**")
                    r1.markdown(issue.category)
                    r2.markdown(custom.get("custom_category") or "—")
                    r3.markdown(agree(issue.category, custom.get("custom_category")))
                    s0, s1, s2, s3 = st.columns([1.1, 2, 2, 0.6])
                    s0.markdown("**Severity**")
                    s1.markdown(severity_badge(issue.severity.value), unsafe_allow_html=True)
                    s2.markdown(severity_badge(custom["custom_severity"]) if custom.get("custom_severity") else "—",
                                unsafe_allow_html=True)
                    s3.markdown(agree(issue.severity.value, custom.get("custom_severity")))
                    st.caption(f"DistilBERT input: “{custom.get('custom_input')}” · trained on synthetic data")
            if analysis.get("model_used"):
                st.caption(f"Gemini model used: `{analysis['model_used']}`"
                           + (" · loaded from demo cache" if analysis.get("from_cache") else ""))

            # --- Evidence lists ---
            with st.expander(f"🖼️ Observed evidence (image) — {len(issue.observed_evidence)}"):
                bullet_list(issue.observed_evidence)
            with st.expander(f"🗣️ User-reported facts — {len(issue.user_reported_facts)}"):
                bullet_list(issue.user_reported_facts)
            with st.expander(f"💭 Inferred information — {len(issue.inferred_information)}"):
                bullet_list(issue.inferred_information)
            with st.expander(f"❓ Missing information — {len(issue.missing_information)}"):
                bullet_list(issue.missing_information)

        audit = analysis["audit"]
        if audit is not None:
            # --- Contradiction audit ---
            with st.container(border=True):
                st.markdown("**⚖️ Contradiction Check**")
                nli_audit = (analysis.get("custom") or {}).get("nli_audit")
                gemini_tab, nli_tab = st.tabs(["Gemini audit (used for complaint)", "MiniLM NLI audit (custom)"])
                with gemini_tab:
                    render_audit(audit)
                    if not audit.audit_passed:
                        st.warning("Some claims are contradicted by the image evidence — they were excluded from the complaint.")
                with nli_tab:
                    if nli_audit is not None:
                        render_audit(nli_audit)
                        st.caption("cross-encoder/nli-MiniLM2-L6-H768 compares each claim against the observed image "
                                   "evidence. Shown for comparison; the complaint uses the Gemini audit.")
                    else:
                        st.caption("MiniLM NLI audit unavailable.")

        complaint = analysis["complaint"]
        if complaint is not None:
            # --- Complaint ---
            with st.container(border=True):
                st.markdown("**📄 Generated Complaint**")
                edited = st.text_area(
                    "Edit before submitting",
                    value=complaint,
                    height=180,
                    key=f"complaint_{v}_{st.session_state.get('run_id', 0)}",
                )
                st.caption("Copy (use the icon at the top-right of the box):")
                st.code(edited, language=None, wrap_lines=True)
                st.download_button(
                    "⬇️ Download Report",
                    data=build_text_report(analysis, edited),
                    file_name=f"civicfix_report_{date.today().isoformat()}.txt",
                    mime="text/plain",
                )

        quality = analysis["quality"]
        if quality is not None:
            # --- Quality score ---
            with st.container(border=True):
                st.markdown("**⭐ Complaint Quality Score**")
                q1, q2 = st.columns(2)
                q1.metric("Before (raw input)", f"{quality.raw_input_score:.0f} / 100")
                q2.metric(
                    "After (AI report)",
                    f"{quality.overall:.0f} / 100",
                    delta=f"{quality.overall - quality.raw_input_score:+.0f}",
                )
                for label, value in (
                    ("Completeness", quality.completeness),
                    ("Specificity", quality.specificity),
                    ("Evidence", quality.evidence),
                    ("Actionability", quality.actionability),
                ):
                    st.progress(int(value), text=f"{label} — {value:.0f} / 100")
                unsupported, total = analysis["unsupported"]
                st.markdown(f"**Unsupported sentences:** {unsupported} / {total}")
                st.caption("Deterministic rule-based score — no AI involved. A sentence is unsupported if it shares "
                           "fewer than 2 content words with any verified fact.")


# ---------- Footer ----------
st.divider()
st.caption(
    "⚠️ Uploaded media is used for analysis only. Avoid submitting sensitive personal information.  \n"
    "Severity and category are AI-assisted estimates, not professional assessments."
)
