# CivicFix — Technical Report

**Project state as of:** 2026-10-27 (after Phase 9 and the post-Phase-9 input-mismatch review; Phase 10 "Final Testing" has not been run yet)
**Repository root:** `C:\Users\khatr\Desktop\rag_project` (stands in for the build plan's `civicfix/` folder)
**Last commit:** `05c87cb Version 1 before ph10` on branch `master`. Uncommitted changes in `app.py`, `src/ai_service.py`, `src/prompts.py`, `src/schemas.py`, `sample_data/voice_samples.md`, `CLAUDE.md`, plus the untracked `.streamlit/`.
**Sources for every number in this report:** the code in the repository, `eval/results/eval_report.json`, `eval/results/eval_cases.csv`, `data/synthetic/*.csv`, the saved model files in `models/saved/`, and the phase log in `CLAUDE.md`. I recomputed the DistilBERT validation accuracies from the saved models on 2026-10-27 while writing this report.

> **Reading note on accuracy.** The report request listed some expected values. Several of them differ from what the code actually does, and this report states the **actual** value every time. The main differences:
> - DistilBERT was trained for **15 epochs**, not 4. Four epochs was the plan's value and the first run, which was later replaced.
> - The real dataset (DATASET_PLAN.md) **has not been collected**. It is deferred until after the 2026-10-28 demo. `data/real/` contains only `.gitkeep`.
> - The photo-grouped train/test split is **planned, not implemented**. The current split is a row-level stratified 80/20 split.
> - The complaint generator receives the **full structured issue as well as** the verified facts. "Verified facts only" is an instruction in the prompt; the code does not enforce it.
> - The MiniLM NLI audit is **computed and displayed but does not feed the complaint**. The complaint uses the Gemini audit.
> - The modality ablation sets are **defined but never executed**.

---

## SECTION 1 — PROJECT OVERVIEW

### 1.1 Problem statement

Civic and campus complaints (a dead streetlight, an overflowing bin, a leaking pipe) are usually reported as short, informal, incomplete messages. They often have no location, no duration and no severity, and are sometimes exaggerated, so maintenance departments cannot route or prioritise them. CivicFix takes any combination of a typed description, a voice recording and a photo and turns it into a structured, evidence-checked complaint: category, severity, department, a formal complaint text and a measurable quality score. Before the complaint is written, CivicFix cross-checks the user's claims against what the photo actually shows. Claims the photo contradicts are excluded from the complaint, and the user must review a photo that does not match.

### 1.2 The 4 research objectives (verbatim from `CIVICFIX_BUILD_PLAN.md`, "Core research objectives")

1. Multimodal structuring pipeline using pre-trained AND custom-trained models
2. Contradiction detection — cross-check user claims against image evidence
3. Hallucination reduction — measure unsupported-sentence rate with/without audit stage
4. Measurable evaluation — modality ablation, classification accuracy, contradiction P/R, deterministic quality score

### 1.3 What makes this different from a generic chatbot — every distinct AI function

A chatbot takes free text and returns free text. CivicFix is a fixed pipeline. Every stage has a typed input, a typed output and a separate implementation:

| # | Function | Implementation | Where |
|---|---|---|---|
| 1 | Speech-to-text (local, no cloud) | OpenAI Whisper `base`, `fp16=False`, CPU | `src/speech_service.transcribe_audio` |
| 2 | Multimodal structured extraction (text + transcript + image → 14-field JSON) | Gemini, system prompt `ANALYSIS_SYSTEM_PROMPT`, temperature 0.1, validated by Pydantic `StructuredIssue` | `src/ai_service.analyze_issue` |
| 3 | Evidence separation: what the image shows vs what the user said vs what is inferred vs what is missing | Four separate lists in the schema (`observed_evidence`, `user_reported_facts`, `inferred_information`, `missing_information`) | `src/schemas.StructuredIssue` |
| 4 | Claim-vs-evidence contradiction audit (LLM) | Gemini, `CONTRADICTION_PROMPT`, temperature 0.0, gives each claim supported / contradicted / unverifiable | `src/ai_service.run_contradiction_audit` |
| 5 | Wrong-photo detection | Same Gemini audit call returns `image_matches_report` + `mismatch_reason` → `ContradictionAudit.image_mismatch` | `src/ai_service.run_contradiction_audit` |
| 6 | Claim-vs-evidence contradiction audit (custom, non-LLM) | Pre-trained cross-encoder `cross-encoder/nli-MiniLM2-L6-H768`, NLI argmax | `src/contradiction.run_nli_audit`, `models/contradiction_model.batch_check` |
| 7 | Category classification (custom) | Fine-tuned `distilbert-base-uncased`, 11 classes | `models/classifier.predict(text, "category")` |
| 8 | Severity classification (custom) | Separately fine-tuned `distilbert-base-uncased`, 4 classes | `models/classifier.predict(text, "severity")` |
| 9 | Grounded complaint generation | Gemini, `COMPLAINT_GENERATION_PROMPT`, temperature 0.3, given the audit's verified facts | `src/ai_service.generate_complaint` |
| 10 | Human-in-the-loop mismatch resolution | If the photo contradicts the text: Re-upload / Edit / Continue. **Continue** re-analyses without the image and writes a flagged, text-priority complaint (`TEXT_PRIORITY_COMPLAINT_PROMPT`) | `app.needs_user_review`, `app.finish_report(..., "text_priority")`, `src/ai_service.generate_text_priority_complaint` |
| 11 | Deterministic quality scoring (no LLM) | Rule-based Completeness / Specificity / Evidence / Actionability + raw-input score | `src/scoring.calculate_quality_score` |
| 12 | Hallucination proxy | Unsupported-sentence counter (word overlap ≥ 2 with any reference fact) | `src/scoring.count_unsupported_sentences`, `eval/metrics.unsupported_sentence_rate` |
| 13 | Synthetic training data generation | Gemini, `GENERATION_PROMPT`, temperature 0.9 | `models/data_generator.py` |
| 14 | Severity re-labelling with a single rubric | Gemini, `RELABEL_PROMPT`, temperature 0.0, batches of 55 rows | `models/relabel_severity.py` |
| 15 | Offline evaluation, Gemini vs custom models | 15 hand-labelled cases; weighted P/R/F1, contradiction P/R/F1, hallucination ablation, quality | `eval/eval_runner.py`, `eval/metrics.py` |

Infrastructure that is not AI but shapes how the AI behaves:
- **Model fallback.** `_generate` tries `GEMINI_MODEL` first, then `GEMINI_FALLBACK_MODELS`. A 429 or 404 skips that model for the rest of the process; a 503 or 504 puts it on a 10-minute cooldown.
- **Demo cache.** `demo_cache/`, keyed by a SHA-1 of text, transcript and image bytes.
- **Eval cache.** `eval/results/cache/`, keyed by a SHA-1 of the case text.

### 1.4 What is NOT in scope for the MVP

- **Video input.** It was added by mistake once and removed; the app has no video uploader.
- **Hindi / Hinglish.** The project switched to English only for the demo and for the real dataset. The UI is English; the synthetic data prompt forbids Hindi words. (The raw-input scorer still contains the Hinglish duration keywords `din`, `ghante`, `hafte` from the original plan.)
- **Real-world data.** The real dataset (real photos, human-verified labels, recordings from 4 speakers) is deferred until after the demo. All custom models are trained on **synthetic** data.
- **Submitting complaints to any authority.** There is no backend, ticketing, database, authentication or multi-user state. The output is an on-screen complaint plus a downloadable `.txt`.
- **Duplicate detection** across complaints. There is no complaint store at all.
- **Geolocation / maps / EXIF.** Location comes only from what the user states or what is clearly visible.
- **Professional severity assessment.** The UI footer says: "Severity and category are AI-assisted estimates, not professional assessments."
- **Fine-tuning of MiniLM.** It is used pre-trained. `models/saved/contradiction/` does not exist, so the code falls back to the Hub model.
- **Running the modality ablation.** `MODALITY_SETS` is defined in `eval/eval_runner.py` but never used.
- **Deployment.** The app runs locally (`streamlit run app.py`) on Windows 11, CPU only.

---

## SECTION 2 — SYSTEM ARCHITECTURE

### 2.1 Full pipeline diagram (as implemented in `app.run_pipeline` / `app.finish_report`)

```
 ┌──────────────────────────── Streamlit UI (app.py) ────────────────────────────┐
 │  st.text_area (text)     st.file_uploader (wav/mp3/m4a/ogg)     st.file_uploader (jpg/jpeg/png/webp)
 └──────────┬──────────────────────────────┬────────────────────────────────┬─────┘
            │ str                          │ UploadedFile                   │ UploadedFile
            │                              ▼                                │
            │               utils.save_temp_audio → Path (temp file)        │
            │                              ▼                                │
            │          speech_service.transcribe_audio(Path)                │
            │          Whisper "base", fp16=False, local CPU                │
            │          (ffmpeg from imageio-ffmpeg if not on PATH)          │
            │                              │ dict{success, transcript,      │
            │                              │      language, error}          │
            │                              ▼                                │
            │                     transcript: str | None                    │
            ▼                              ▼                                ▼
 ┌─────────────────────── demo cache lookup (sidebar toggle, default ON) ───────────────────────┐
 │ key = sha1(text \0 transcript \0 image_bytes)[:16] → demo_cache/<key>.json                   │
 │ HIT → load issue/audit (skip both Gemini calls below)        MISS → continue                 │
 └──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                ▼
                     utils.load_image_pil → PIL.Image (→ .convert("RGB") → JPEG bytes)
                                                ▼
   ai_service.analyze_issue(text, transcript, image)  ── Gemini call #1 ─ T=0.1, system=ANALYSIS_SYSTEM_PROMPT
   parse JSON (strip ``` fences) → StructuredIssue (Pydantic); 2 attempts
                                                │ StructuredIssue
                    ┌───────────────────────────┼──────────────────────────────┐
                    ▼                           ▼                              │
   app.run_custom_models (never raises)    ai_service.run_contradiction_audit ── Gemini call #2 ─ T=0.0
   ├ DistilBERT category (11 classes)        claims = user_reported_facts + inferred_information
   ├ DistilBERT severity (4 classes)         evidence = observed_evidence
   │   input = text + transcript             → flags[] + image_matches_report
   │   (or issue.problem if image-only)      → ContradictionAudit (verified / contradicted /
   └ MiniLM NLI audit (run_nli_audit)          unverifiable / image_mismatch / audit_passed)
     → ContradictionAudit (DISPLAY ONLY)                │
                    │                                   ▼
                    │                  app.needs_user_review(result)?
                    │            has_image AND (contradicted_claims OR image_mismatch)
                    │                    │ NO                              │ YES
                    │                    ▼                                 ▼
                    │     finish_report(result,"verified")      Review card: ❌ claims + reason +
                    │     ai_service.generate_complaint         "What the photo actually shows"
                    │     ── Gemini call #3 ─ T=0.3             ┌──────────┬──────────────┬──────────────┐
                    │     inputs: issue JSON + verified_facts   │Re-upload │Edit descr.   │Continue with │
                    │                    │                      │photo     │              │my description│
                    │                    │                      │(clears   │(clears       └──────┬───────┘
                    │                    │                      │image +   │results)             ▼
                    │                    │                      │results)  │   finish_report(result,"text_priority")
                    │                    │                      └──────────┘   analyze_issue(text, transcript, None)
                    │                    │                                     ── Gemini (re-classify, no image)
                    │                    │                                     generate_text_priority_complaint
                    │                    │                                     ── Gemini T=0.3 (user facts only)
                    │                    ▼                                                 ▼
                    │      scoring.calculate_quality_score(issue, audit, text, has_image, has_audio) → QualityScore
                    │      scoring.count_unsupported_sentences(complaint, reference) → (unsupported, total)
                    │         reference = audit.verified_facts (+ user_reported_facts in text_priority mode)
                    ▼                                    ▼
 ┌──────────────────────────────────── Results column ──────────────────────────────────────────┐
 │ Voice transcript · ⚠ Flag (text_priority) · Issue card (problem, category, severity badge,    │
 │ department, reason, location, duration, confidence, action) · Before vs After · Gemini vs     │
 │ DistilBERT table · 4 evidence expanders · Contradiction tabs (Gemini | MiniLM) · editable      │
 │ complaint + copy box + ⬇ Download Report (.txt) · Quality score card                           │
 └────────────────────────────────────────────────────────────────────────────────────────────────┘
 Sidebar: demo-cache toggle · Architecture (st.code) · Evaluation Summary (reads eval/results/eval_report.json)
```

Gemini call count per analysis: a matched input costs 3 calls (analysis, audit, complaint). A mismatched input costs 2 calls before the user decides, plus 2 more if they choose Continue (re-analysis and text-priority complaint). With a demo-cache hit, the cost is 0 calls.

> The sidebar `ARCHITECTURE` string in `app.py` says "User Claims + Evidence → MiniLM NLI → Contradiction Audit" followed by "Verified Facts → Gemini → Complaint". That reads as if MiniLM feeds the complaint. In the code it does not: the complaint uses the **Gemini** audit, and the MiniLM audit is shown in a separate tab for comparison only (the tab caption says so).

### 2.2 Every module

**`app.py` (703 lines).** This is the Streamlit front end and the orchestrator. It defines:
- `SEVERITY_COLORS`: Low `#2e7d32`, Medium `#fbc02d` (dark text), High `#ef6c00`, Critical `#c62828`.
- `friendly_error`, which maps API-key, 429/quota, model-404 and 503 errors to plain messages, so no stack trace reaches the user.
- `run_custom_models`, which lazily imports DistilBERT and MiniLM and never raises.
- `build_text_report`, the downloadable plain-text report. It uses the user-**edited** complaint text and adds a `FLAG:` line in text-priority mode.
- `render_eval_summary`, a sidebar table built from `eval/results/eval_report.json`.
- The demo cache helpers `_cache_path`, `load_cached_gemini` and `save_cached_gemini`.
- `needs_user_review`, `finish_report` and `run_pipeline`.
- Session-state management: `input_version` and `image_version` are suffixed onto widget keys so Reset and Re-upload can clear `file_uploader` widgets; `run_id` refreshes the complaint `text_area`.
- The full layout: a 2:3 column split (inputs | results), an Analyze button that is disabled when all inputs are empty, Reset, and a footer disclaimer.

All pipeline state lives in `st.session_state["analysis"]`, a dict with these keys: `has_text, has_audio, has_image, transcription, issue, audit, complaint, quality, unsupported, custom, model_used, original_input, needs_review, report_mode, ctx, issue_original, error` (plus `from_cache` on a cache hit).

**`config.py` (44 lines).** It loads `.env` via `python-dotenv`, which does not override variables already set in the environment. It exposes:
- `GEMINI_API_KEY`.
- `GEMINI_MODEL` (default `"gemini-3.5-flash"`).
- `GEMINI_FALLBACK_MODELS`, a comma list from the environment; default `gemini-3.8-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.7-flash`.
- `CATEGORIES`, 11 values: Electrical, Plumbing / Water, Sanitation, Waste Management, Infrastructure, Security, Cleaning / Housekeeping, Furniture / Facilities, Internet / Network, Transport, Other.
- `SEVERITY_LEVELS`: Low, Medium, High, Critical.
- `DEPARTMENTS`, a category→department map (e.g. Electrical → Electrical Maintenance, Internet / Network → IT Support). It is defined but not referenced by the pipeline. Gemini chooses the department itself, and its department strings do not always match this map.

**`src/ai_service.py` (195 lines).** This module holds all Gemini access, through the `google-genai` SDK:
- `_get_client()` builds one `genai.Client` with a 120 000 ms HTTP timeout.
- `_generate(contents, config)` is the fallback wrapper. It tries each candidate model. On a `genai_errors.APIError` with code 404 or 429, it adds the model to `_exhausted_models`; on 503 or 504, it sets `_cooldown_until[model] = now + 600 s`. Any other error code is re-raised. It records `last_model_used`.
- `_parse_json_response` strips Markdown code fences.
- `analyze_issue(text, transcript, image_pil) -> StructuredIssue` makes 2 attempts.
- `run_contradiction_audit(issue) -> ContradictionAudit` makes 2 attempts and falls back to a permissive audit if both fail.
- `generate_complaint(issue, audit) -> str` makes a single attempt.
- `generate_text_priority_complaint(issue, audit) -> str` makes a single attempt and excludes `observed_evidence` and `inferred_information` from the issue JSON it sends.

**`src/speech_service.py` (60 lines).** Local speech-to-text with `openai-whisper`:
- `_ensure_ffmpeg()` checks whether `ffmpeg` is on PATH. If not, it copies the `imageio-ffmpeg` bundled binary (ffmpeg 7.1) once to `%TEMP%\civicfix_ffmpeg\ffmpeg.exe` and prepends that directory to `PATH`; Whisper shells out to `ffmpeg`.
- `_get_model()` lazily loads `whisper.load_model("base")` (≈139 MB in `~/.cache/whisper`).
- `transcribe_audio(audio_path: Path) -> dict` calls `model.transcribe(str(path), fp16=False)` with automatic language detection and returns `{"success": bool, "transcript": str|None, "language": str|None, "error": str|None}`. It never raises; failures are logged and returned as `success=False`.
- In the UI, an empty transcript (silent audio) is also treated as a failure.

**`src/schemas.py` (68 lines).** These are the Pydantic v2 models that every AI output must pass before it reaches the UI (the full schema is in §7.2):
- `SeverityLevel`, a `str` Enum with the case-sensitive values `Low/Medium/High/Critical`. "high" is rejected.
- `ContradictionFlag`.
- `StructuredIssue`, with `confidence` constrained to [0, 1].
- `ContradictionAudit`, including `image_mismatch` and `mismatch_reason` with defaults, so older cached JSON still validates.
- `QualityScore`, with every field constrained to [0, 100].
- `CivicFixReport`, an aggregate report type. It is defined but **not instantiated** anywhere in the current code.

**`src/prompts.py` (125 lines).** This module holds every runtime prompt: `ANALYSIS_SYSTEM_PROMPT`, `COMPLAINT_GENERATION_PROMPT`, `TEXT_PRIORITY_COMPLAINT_PROMPT` and `CONTRADICTION_PROMPT`. It also has `build_analysis_prompt()`, which formats the category list into the analysis prompt. It also passes `severity_levels=`, but the template has no `{severity_levels}` placeholder, so that argument is unused and harmless. The full text of each prompt is quoted in Section 7. The generator prompt and the re-labelling prompt live in their own scripts.

**`src/scoring.py` (125 lines).** Pure functions with no LLM and no randomness:
- `score_raw_input`
- `score_completeness`, `score_specificity`, `score_evidence`, `score_actionability`
- `calculate_quality_score -> QualityScore`
- `count_unsupported_sentences -> (unsupported, total)`

Formulas are in Section 10 and the algorithm in Section 9.

**`src/contradiction.py` (39 lines).** `run_nli_audit(issue) -> ContradictionAudit`:
- Claims are `issue.user_reported_facts + issue.inferred_information`.
- The premise is `" ".join(issue.observed_evidence)`, or the literal string `"No image evidence available."`.
- It calls `models.contradiction_model.batch_check` and builds a `ContradictionAudit` in the same shape as the Gemini audit. The per-claim reason is `"NLI score: <max logit>"`, and `image_mismatch` is not set.
- The module docstring says "Hybrid… Returns merged ContradictionAudit", but the function does **not** merge with the Gemini audit; it returns the NLI-only audit.

**`models/data_generator.py` (89 lines).** Synthetic data generator, run with `python -m models.data_generator`:
- It builds its own `genai.Client` with a 180 000 ms timeout and calls `GEMINI_MODEL` directly, **without** the fallback wrapper, using `GENERATION_PROMPT` at temperature 0.9.
- `generate_dataset(total=200, batch_size=20)` sleeps 1.5 s between batches and skips any batch that fails.
- Filters: category ∈ `CATEGORIES`, severity ∈ `SEVERITY_LEVELS`, `dropna`, de-duplication on `raw_complaint`.
- Output: `data/synthetic/complaints.csv`. **Running it overwrites the current 220-row, re-labelled CSV.**

**`models/relabel_severity.py` (72 lines).** This script was added in Phase 7, beyond the plan. It re-labels every synthetic row's severity with one rubric (`RELABEL_PROMPT`):
- It goes through `ai_service._generate` at temperature 0.0, in batches of `BATCH = 55` rows, so 220 rows take 4 calls.
- It makes 2 attempts per batch.
- It refuses to write the CSV if any row is still unlabelled.
- It keeps the old label in `severity_original`.

**`models/classifier.py` (152 lines).** DistilBERT fine-tuning and inference:
- `load_data` fits a `LabelEncoder` on the label column.
- `train(label_col, csv_path, epochs)` runs a stratified 80/20 split, `DistilBertTokenizerFast` (`truncation=True, padding=True, max_length=128`), `DistilBertForSequenceClassification` and the HF `Trainer`. It saves the model and tokenizer, the label map JSON, and prints a `classification_report`.
- `predict(text, label_col) -> str | None` uses a cached `pipeline("text-classification", device=-1)`, parses `LABEL_k`, then looks up `labels[str(k)]`.
- CLI: `--train`, `--data` (default `data/synthetic/complaints.csv`), `--epochs` (default **15**).

**`models/contradiction_model.py` (76 lines).** The MiniLM NLI wrapper around `sentence_transformers.CrossEncoder`:
- `MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"`, with a local override at `models/saved/contradiction/`, which does not exist.
- `check_contradiction(claim, evidence) -> {"label","score","status"}` handles a single pair.
- `batch_check(claims, evidence_text) -> list[{"claim","label","score","status"}]` handles a batch.
- Label map `{0: "contradiction", 1: "entailment", 2: "neutral"}`, confirmed against the model's `id2label`. Status map: entailment→supported, neutral→unverifiable, contradiction→contradicted.
- It imports `pandas` but never uses it.

**`eval/eval_runner.py` (268 lines).** The evaluation suite, run with `python -m eval.eval_runner`. It contains:
- `TEST_CASES` (15 hand-labelled cases).
- `MODALITY_SETS` (5 sets, unused).
- `normalize_category()`, which maps a Gemini "Main / Sub" string to the config category it starts with, longest category first.
- `_gemini_outputs()`, which runs or loads the per-case cache with issue, audit, complaint, `complaint_no_audit` and model_used.
- `run_eval()`, which writes `eval/results/eval_report.json` and `eval/results/eval_cases.csv`.

**`eval/metrics.py` (62 lines).**
- `classification_metrics(y_true, y_pred, labels)` returns weighted P/R/F1 (`zero_division=0`) plus the full `classification_report(output_dict=True)`.
- `contradiction_metrics(y_true, y_pred)` returns TP/FP/FN and P/R/F1 for the positive class `"contradicted"`.
- `unsupported_sentence_rate(complaint, facts)` is the same algorithm as `scoring.count_unsupported_sentences`, but returns a dict with the rate.

**Also present:**
- `src/utils.py`: `image_to_base64` (unused), `save_temp_audio(uploaded_file) -> Path` (a `NamedTemporaryFile(delete=False)` with the original suffix), and `load_image_pil(uploaded_file) -> PIL.Image`.

### 2.3 Data flow — types in and out

| Function | Input (types) | Output (type) |
|---|---|---|
| `utils.save_temp_audio` | `streamlit.UploadedFile` | `pathlib.Path` (temp file, deleted by the caller in `finally`) |
| `speech_service.transcribe_audio` | `Path` | `dict[str, bool \| str \| None]`: `success: bool`, `transcript: str\|None`, `language: str\|None`, `error: str\|None` |
| `utils.load_image_pil` | `UploadedFile` | `PIL.Image.Image` (any mode; RGBA for the 3 sample PNGs) |
| `ai_service.analyze_issue` | `text: str\|None`, `transcript: str\|None`, `image_pil: PIL.Image\|None` | `StructuredIssue`; raises `RuntimeError` after 2 failed attempts |
| `ai_service.run_contradiction_audit` | `StructuredIssue` | `ContradictionAudit` (never raises; fallback audit after 2 failures) |
| `ai_service.generate_complaint` | `StructuredIssue`, `ContradictionAudit` | `str` (3–5 sentences requested) |
| `ai_service.generate_text_priority_complaint` | `StructuredIssue`, `ContradictionAudit` | `str` |
| `ai_service._generate` | `contents: list[str \| types.Part]`, `config: types.GenerateContentConfig` | `GenerateContentResponse`; raises the last `APIError` if every model fails |
| `classifier.predict` | `text: str`, `label_col: "category" \| "severity"` | `str` (a label from `models/saved/labels/*.json`) or `None` if the model directory is missing |
| `contradiction_model.batch_check` | `claims: list[str]`, `evidence_text: str` | `list[dict]` with `claim: str, label: str, score: float (raw logit), status: str` |
| `contradiction.run_nli_audit` | `StructuredIssue` | `ContradictionAudit` |
| `app.run_custom_models` | `issue: StructuredIssue`, `text: str`, `transcript: str\|None` | `dict`: `custom_category: str\|None, custom_severity: str\|None, custom_input: str, nli_audit: ContradictionAudit\|None, custom_error: str\|None` |
| `scoring.calculate_quality_score` | `StructuredIssue`, `ContradictionAudit`, `raw_text: str`, `has_image: bool`, `has_audio: bool` | `QualityScore` |
| `scoring.count_unsupported_sentences` | `complaint: str`, `verified_facts: list[str]` | `tuple[int, int]` = (unsupported, total) |
| `app.needs_user_review` | `result: dict` | `bool` |
| `app.finish_report` | `result: dict` (mutated in place), `mode: "verified" \| "text_priority"` | `None`; sets `complaint`, `quality`, `unsupported`, `report_mode`, `needs_review=False`, and in text-priority mode `issue` / `issue_original` |
| `app.run_pipeline` | `text_input: str`, `audio_file: UploadedFile\|None`, `image_file: UploadedFile\|None`, `use_cache: bool` | `dict` (the analysis state described in §2.2) |
| `app.build_text_report` | `analysis: dict`, `complaint: str` (edited) | `str` (plain-text report) |
| `metrics.classification_metrics` | `y_true: list[str]`, `y_pred: list[str]`, `labels: list[str]` | `dict` with `weighted_precision, weighted_recall, weighted_f1: float`, `per_class: dict` |
| `metrics.contradiction_metrics` | `y_true: list[str]`, `y_pred: list[str]` | `dict` with P/R/F1 `float`, TP/FP/FN `int` |
| `metrics.unsupported_sentence_rate` | `complaint: str`, `verified_facts: list[str]` | `dict` with `unsupported_sentences: int, total_sentences: int, unsupported_rate: float` |

### 2.4 Why Streamlit rather than Flask or FastAPI

- **The product is a single-user, interactive demo, not a service.** A research prototype needs input widgets, intermediate results and plots, and it must stay in one Python process with the models. Streamlit renders `st.file_uploader`, `st.audio`, `st.image`, `st.status` (live pipeline steps), `st.tabs`, `st.metric`, `st.progress`, `st.download_button` and an editable `st.text_area` directly from Python. Flask or FastAPI would need a separate HTML/JS front end, upload handling and a JSON API. That is roughly double the code with no research value.
- **The ML models live in the same process.** Whisper (≈139 MB), two DistilBERT models (257 MB each on disk) and MiniLM are loaded once into module-level globals (`_model`, `_loaded_models`) and reused across reruns. There is no serialisation boundary.
- **State is per session.** `st.session_state` holds the Pydantic objects themselves, so a multi-step decision (the mismatch review → Continue) needs no database and no request routing.
- **Testability without a browser.** `streamlit.testing.v1.AppTest` drove every phase's regression tests headlessly.
- **Accepted downsides.** The whole script re-runs on every interaction (handled with session state and widget-key versioning). There is no concurrency model or REST API. And `file_uploader` cannot be cleared programmatically, which is why widget keys carry `input_version` and `image_version`.

---

## SECTION 3 — TECHNOLOGY STACK WITH JUSTIFICATION

Installed versions (from `CLAUDE.md`): Python 3.13.7, streamlit 1.66.1, google-genai 2.33.0, pydantic 2.14.1, torch 2.14.1 (CPU), transformers 5.20.2, sentence-transformers 6.2.0, datasets 5.1.0, evaluate 0.4.6, openai-whisper 20250625, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, accelerate 1.15.0, imageio-ffmpeg 0.6.0.

> The rows below say which comparisons were **measured** in this project and which are **design reasoning**. Only Gemini vs DistilBERT (classification) and Gemini vs MiniLM (contradiction) were measured. The other alternatives were not benchmarked here.

### Google Gemini API — vs GPT-4o, vs Claude
- **What it is:** Google's multimodal LLM family, called through the `google-genai` SDK (`client.models.generate_content`).
- **Role here:**
  - The single multimodal reasoning engine: image understanding and structured JSON extraction, the claim-by-claim contradiction audit, wrong-photo detection and complaint writing.
  - Offline, it also generated the synthetic data and re-labelled severity.
- **Model strings:**
  - Configured primary: `gemini-3.5-flash`.
  - Fallbacks: `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.7-flash`.
  - Actually served the 15-case eval and every demo-cache entry: `gemini-3.5-flash-lite`.
  - Synthetic data: `gemini-3.5-flash` (80 rows) and `gemini-3.1-flash-lite` (140 rows).
- **Why Gemini:**
  1. **A free API tier** that accepts image input. The team had no budget; billing was not enabled (user decision). GPT-4o and Claude have no comparable free API tier.
  2. **Image and text go in the same request** (`types.Part.from_bytes(..., mime_type="image/jpeg")` alongside text parts), so one call can separate `observed_evidence` from `user_reported_facts`.
  3. **The build plan specified it** ("Best multimodal capability, handles Hinglish"), and the project began with Hinglish in scope.
- **Not benchmarked:** GPT-4o and Claude were not compared in this project, so no quality claim against them is made.
- **Trade-off accepted:** the free tier allows 20 `generate_content` requests per day per model. That forced the fallback chain, the demo cache and the eval cache. Several models were retired (404: `gemini-2.0-flash`, `gemini-2.5-flash`) or overloaded (503/504) during development, and the model that serves a request can change, which reduces reproducibility.

### Whisper — vs Google Speech-to-Text, vs Azure Speech
- **What it is:** OpenAI's open-weights encoder-decoder speech recognition model. Here the `base` checkpoint (74 M parameters, multilingual, automatic language detection) runs locally with `fp16=False` on CPU.
- **Role here:** voice recording → transcript, which is merged with the typed text as `VOICE TRANSCRIPT:` in the analysis prompt.
- **Why:**
  1. **Privacy.** Audio never leaves the machine; only the transcript is sent to Gemini. DATASET_PLAN §8.1 uses this in the consent wording.
  2. **No API key, no quota, no cost.** Google and Azure STT need cloud accounts, billing and per-minute charges, and Gemini's quota was already the bottleneck.
  3. **Works offline** with any ffmpeg-readable format (wav/mp3/m4a/ogg).
  4. **Language auto-detection**, needed while Hinglish was in scope.
- **Why `base`:** a speed/size compromise for CPU (≈139 MB). `tiny` is less accurate; `small` and larger are slower on CPU.
- **Not measured:** no word error rate has been measured yet. The demo clips are Windows TTS voices; DATASET_PLAN §9 plans a WER per speaker, gender and environment.

### DistilBERT — vs BERT-base, vs RoBERTa, vs zero-shot Gemini classification
- **What it is:** a 6-layer Transformer encoder distilled from BERT-base. With the classification head, the saved category model has 66,961,931 parameters.
- **Role here:** two independent fine-tuned classifiers (category, 11 classes; severity, 4 classes) as the custom-trained comparison point against Gemini.
- **Why:**
  1. **Trains on CPU** in minutes on 176 rows (no GPU on this machine). Fine-tuning BERT-base (≈110 M) or RoBERTa-base (≈125 M) would take about 2× the compute for an expected small gain on 220 short synthetic sentences.
  2. **Fast local inference** with no API call, no quota and deterministic output.
  3. **A real "custom-trained model" for objective 1.** Zero-shot Gemini classification is already what the Gemini baseline does (Gemini assigns `category` and `severity` in `analyze_issue`), so DistilBERT is the *trained* counterpart in the comparison.
- **Uncased** because user complaints are frequently lowercase or inconsistently cased.
- **Trade-off (measured):**
  - On the 15 eval cases DistilBERT reached 53% category accuracy vs Gemini's 100%.
  - It reached 67% severity accuracy vs Gemini's 73%.
  - The gap is mainly the synthetic training data (Section 13), not the architecture choice, but this report does not prove that.

### MiniLM cross-encoder NLI — vs BERT NLI, vs GPT-4 contradiction check, vs rule-based
- **What it is:** `cross-encoder/nli-MiniLM2-L6-H768`, a MiniLMv2 model. Its config reports `model_type: roberta`, 6 layers, hidden size 768, 82,120,707 parameters and `max_length` 512, with a 3-way NLI head (`id2label` = {0: contradiction, 1: entailment, 2: neutral}). Per its model card, it was trained on the SNLI and MultiNLI datasets.
- **Role here:** a second, non-LLM contradiction audit. Premise = image evidence text; hypothesis = user claim.
- **Why:**
  1. **Purpose-built for entailment and contradiction**; no training data is needed (plan: "no training needed").
  2. **Small and local.** 6 layers instead of 12–24 for BERT-base or RoBERTa-large NLI models, so it runs on CPU in a fraction of a second per claim batch.
  3. **An independent signal** from Gemini, which avoids "LLM grading itself". A GPT-4 checker would be another paid LLM call with the same opacity and quota problems.
  4. **Rule-based alternatives** (keyword or negation matching) cannot catch paraphrase or scale contradictions such as "10-foot craters" vs "small pothole".
- **Trade-off (measured):**
  - It only sees text: the Gemini-written `observed_evidence`, never the image.
  - It is aggressive. During Phase 7 testing, "light broken 3 days" vs "streetlight fixture visible" was labelled contradiction.
  - On the eval: P = 0.50, R = 0.33.

### Pydantic v2 — vs dataclasses, vs marshmallow
- **What it is:** a runtime data-validation library (Rust core in v2).
- **Role here:** every Gemini JSON output is validated into `StructuredIssue` / `ContradictionFlag` / `ContradictionAudit` before use. `QualityScore` bounds the scores, and `model_dump_json` / `model_validate` serialise the demo and eval caches.
- **Why:**
  1. **Validation of untrusted LLM output.** A severity outside the Enum (e.g. "Extreme" or "high") or a confidence of 1.5 raises `ValidationError`, which triggers the retry. Dataclasses do no runtime validation.
  2. **One class gives parse + validate + serialise.** `model_validate`, `model_dump(mode="json")` and `model_dump_json` are used throughout; marshmallow needs separate schema classes.
  3. **Defaults keep old cache files loading.** `image_mismatch: bool = False` and `mismatch_reason: Optional[str] = None` were added later, and old cache JSON still validates.
- **Trade-off:** `ContradictionFlag.status` is a plain `str`, so an invalid status from Gemini would pass validation. It would then fall into none of the three lists.

### Streamlit — vs Flask + React, vs Gradio
- **What it is:** a Python framework that renders a web UI from a top-to-bottom script.
- **Role here:** the entire UI (Section 2.4).
- **Why not Flask + React:** two codebases and an API layer, which add nothing to the research objectives.
- **Why not Gradio:** Gradio is excellent for a single input→output function. CivicFix needs a multi-card dashboard: tabs, metrics, progress bars, a sidebar summary, an editable complaint, a download button, and a stateful 3-button review step that only appears on a mismatch and changes what happens next. Streamlit's `session_state` plus layout primitives (`columns`, `container(border=True)`, `tabs`, `status`) handle this layout and state more directly.
- **Trade-off:** full-script reruns, and a file watcher that logs harmless `torchvision` tracebacks from transformers' lazy modules. The watcher is disabled in `.streamlit/config.toml` (`fileWatcherType = "none"`), so the app must be restarted after code edits.

### sentence-transformers CrossEncoder — vs bi-encoder, vs cosine similarity
- **What it is:** the `CrossEncoder` class feeds both texts together through one Transformer and outputs a classification head (here 3 logits).
- **Why a cross-encoder:** contradiction is a **relation between two texts**. It needs token-level attention across premise and hypothesis: "light *is* working" vs "light *is not* working" differ in one token.
- **Why not a bi-encoder or cosine similarity:** a bi-encoder encodes each text separately into a vector and compares them with cosine similarity. That measures topical **similarity**, and a claim and its negation are highly similar. Cosine similarity cannot tell "supported" from "contradicted"; it has no notion of direction or negation.
- **Trade-off:** a cross-encoder must run once per (premise, claim) pair, so nothing can be pre-computed. With fewer than 10 claims per complaint, that cost does not matter.

---

## SECTION 4 — DATASET

### 4.1 Synthetic dataset — how it was generated

**Script:** `models/data_generator.py`, run with `python -m models.data_generator`.

**Exact prompt (`GENERATION_PROMPT`, verbatim; `{n}`, `{categories}`, `{severity_levels}` are filled at run time):**
```
Generate {n} realistic civic complaint examples for a college/hostel environment.

For each example, provide:
1. raw_complaint: informal English text (1-2 sentences, may include typos, abbreviations, vague language). English only — no Hindi or Hinglish words.
2. category: one of {categories}
3. severity: one of {severity_levels}
4. has_contradiction: true if the complaint contains an exaggerated or unverifiable claim, false otherwise

Return ONLY a JSON array. No markdown.

Example entry:
{{
  "raw_complaint": "light not working outside hostel since 3 days",
  "category": "Electrical",
  "severity": "High",
  "has_contradiction": false
}}
```
(Deviation from the plan: "Hinglish" was removed and the example entry was made English.)

**Parameters:**
- `temperature=0.9` for diversity.
- `n = batch_size = 20` per call; `total=200` → 10 batches; `time.sleep(1.5)` between batches.
- HTTP timeout 180 s.
- Fence-stripping JSON parse.
- Post-filters: category ∈ the 11 `CATEGORIES`, severity ∈ the 4 `SEVERITY_LEVELS`, drop NaN in `raw_complaint/category/severity`, drop duplicate `raw_complaint`.

**Run history (from `CLAUDE.md`):**
- The first run on `gemini-3.5-flash` completed 4 batches; the rest failed with one 503, then a 429 (daily quota) → 80 rows (`data/synthetic/complaints_batch1.csv`, 80 × 4).
- The remaining batches ran on `gemini-3.1-flash-lite` via `$env:GEMINI_MODEL` → 140 rows (`complaints_batch2.csv`, 140 × 4).
- The two parts were merged with the same filters → 220 rows, plus a `source_model` column.

**Severity re-labelling (Phase 7):**
- **Why:** the generator's severity labels were inconsistent ("chair missing leg" = Low vs "chair loose" = Medium). A TF-IDF + logistic regression model reached only 0.49 accuracy in 5-fold cross-validation, and DistilBERT severity plateaued around 0.50.
- **What was done:** `models/relabel_severity.py` re-labelled all 220 rows with a single rubric. Temperature 0.0, 4 calls of 55 rows, through the fallback wrapper.
- **Result:** **105 of 220 labels (47.7%) changed.** The old label is kept in `severity_original`; the pre-relabel backup is `data/synthetic/complaints_before_relabel.csv` (220 × 5). TF-IDF + LR cross-validation accuracy rose to 0.62.

Exact re-labelling prompt (`RELABEL_PROMPT`, verbatim):
```
You are labelling civic complaints from a college/hostel by SEVERITY.
Apply these definitions strictly and consistently — the same kind of problem must get the same label:

- Low: Minor inconvenience, no safety impact, few users affected (e.g. one flickering light, a wobbly chair, slow but working wifi, dusty corridor).
- Medium: Meaningful inconvenience, maintenance needed, not urgent (e.g. wifi down in one area, leaking tap, broken chair, bin not emptied, bus late).
- High: Significant impact on safety, accessibility, sanitation, or essential services (e.g. dark corridor/street at night, overflowing garbage or toilets, water leak flooding a floor, broken gate lock, no water supply).
- Critical: Immediate danger or serious hazard requiring urgent response (e.g. exposed live wires or sparks, fire, gas smell, sewage overflow, structural collapse, contaminated drinking water making people ill, blocked emergency access).

Judge the underlying problem, not the writer's tone: ignore exaggerations ("100 days", "dead body smell", "whole college") when deciding severity.

Return ONLY a JSON array of objects {{"id": <int>, "severity": "Low" | "Medium" | "High" | "Critical"}}, one per complaint, same ids.

COMPLAINTS:
{rows}
```

Original → re-labelled severity (crosstab from `data/synthetic/complaints.csv`):

| original \ new | Critical | High | Low | Medium | Total |
|---|---|---|---|---|---|
| Critical | 11 | 11 | 0 | 1 | 23 |
| High | 12 | 33 | 0 | 18 | 63 |
| Low | 0 | 1 | 32 | 20 | 53 |
| Medium | 1 | 18 | 23 | 39 | 81 |
| **Total** | **24** | **63** | **55** | **78** | **220** |

**Sample rows (random_state=1):**

| raw_complaint | category | severity | has_contradiction |
|---|---|---|---|
| The campus internet is the slowest in the entire world, can't even open Google. | Internet / Network | Medium | True |
| common room tv area is very dusty and couches have food stains | Cleaning / Housekeeping | Low | False |
| There are cockroaches in the common room area near the sofa. | Sanitation | Medium | False |
| The tap in the common bathroom has been leaking for a week and wasting so much water. | Plumbing / Water | Medium | False |
| fire extinguisher near room 310 expired two years ago and pressure gauge shows red zone | Security | High | False |
| food waste bin behind mess is completely empty but overflowing with tons of trash | Waste Management | High | True |

### 4.2 Real dataset — how it is planned to be collected (NOT YET COLLECTED)

**Status: not collected.** It was deferred by user decision on 2026-10-26, with the demo on 2026-10-28 as the priority. `data/real/` contains only `.gitkeep`. No `tools/*.py` scripts exist yet, and there is no `data/real/train.csv` or `test.csv`. Everything in this subsection is the **plan** from `DATASET_PLAN.md`.

- **Photo sources, in order of preference (§4.1):**
  1. The team's own phone photos of campus, hostel or neighbourhood.
  2. Free-licence sites: Wikimedia Commons (CC0 / CC-BY), Unsplash, Pexels.
  3. A handful of images from public research datasets, after checking each licence: TACO (litter), RDD2022 (road damage, includes Indian roads), Kaggle pothole / garbage / water-leak sets.
  - Scraping Google Images or news sites is explicitly forbidden.
  - Each non-own photo gets a line in `images/<category_folder>/SOURCES.txt`: `<filename> | <source URL> | <licence>`.
- **Photo rules:**
  - `.jpg`/`.png`, ≥ 640 px on the short side.
  - One clear problem per photo.
  - No faces, licence plates, ID cards or room name-tags (crop or blur).
  - A mix of day/night, close-up/wide and indoor/outdoor, with a severity spread per category.
- **Categories per member (§0):**

| ID | Role | Categories owned |
|---|---|---|
| M1 | Lead (runs Claude steps, merges files), speaker | Electrical · Security · Transport |
| M2 | Member, speaker | Infrastructure · Plumbing / Water · Internet / Network |
| F1 | Member, speaker | Waste Management · Sanitation · Other |
| F2 | Member, speaker, **second reviewer** of the test set | Cleaning / Housekeeping · Furniture / Facilities |

- **Naming convention:** `<prefix>_<NNN>.jpg`, numbered from 001 within each category. Prefixes: `el` Electrical, `se` Security, `tr` Transport, `in` Infrastructure, `pw` Plumbing / Water, `nw` Internet / Network, `wm` Waste Management, `sn` Sanitation, `ot` Other, `ch` Cleaning / Housekeeping, `ff` Furniture / Facilities.
  - Folders: `data/real/images/<electrical|security|transport|infrastructure|plumbing_water|internet_network|waste_management|sanitation|other|cleaning_housekeeping|furniture_facilities>/`.
  - Audio: `data/real/audio/<ID>/<ID>_<record_id>_<quiet|noisy>.m4a`, e.g. `F1_pw_004_informal_quiet.m4a`; the free-speech suffix is `_free`.
- **Planned size (recommended):**
  - 130–160 photos (≈35–40 per member).
  - ≈650 texts.
  - ≥ 130 contradiction examples.
  - 80–100 voice clips (20–25 per member).
  - A 120-text human-verified test set.
- **Balance rules:** every category ≥ 20 texts; every severity ≥ 40 texts; no category above ≈15%; each member ≥ 4 clearly High/Critical photos.

### 4.3 Dataset statistics (actual, `data/synthetic/complaints.csv`)

- **Total rows: 220.** Columns: `raw_complaint, category, severity, has_contradiction, source_model, severity_original`.
- **Text length:** 8–21 words per row, mean 14.6. There are 0 empty texts and 0 Hindi words (checked in Phase 6).
- **Source model:** `gemini-3.1-flash-lite` 140, `gemini-3.5-flash` 80.

**Per category:**

| Category | Rows |
|---|---|
| Plumbing / Water | 30 |
| Electrical | 27 |
| Security | 27 |
| Furniture / Facilities | 26 |
| Infrastructure | 23 |
| Cleaning / Housekeeping | 20 |
| Internet / Network | 18 |
| Waste Management | 15 |
| Transport | 12 |
| Sanitation | 11 |
| Other | 11 |

**Per severity (after re-labelling):** Medium 78 · High 63 · Low 55 · Critical 24.

**has_contradiction:** False 166 · True 54 (24.5%). Note: in the synthetic data `has_contradiction` means "the text contains an exaggerated or unverifiable claim". It is a property of the text alone; there is no image. The custom models do not use this column for training.

**Category × severity:**

| Category | Critical | High | Low | Medium | Total |
|---|---|---|---|---|---|
| Cleaning / Housekeeping | 1 | 1 | 12 | 6 | 20 |
| Electrical | 9 | 6 | 10 | 2 | 27 |
| Furniture / Facilities | 0 | 2 | 8 | 16 | 26 |
| Infrastructure | 4 | 10 | 7 | 2 | 23 |
| Internet / Network | 0 | 0 | 3 | 15 | 18 |
| Other | 0 | 1 | 5 | 5 | 11 |
| Plumbing / Water | 4 | 13 | 0 | 13 | 30 |
| Sanitation | 4 | 5 | 0 | 2 | 11 |
| Security | 1 | 14 | 7 | 5 | 27 |
| Transport | 1 | 0 | 1 | 10 | 12 |
| Waste Management | 0 | 11 | 2 | 2 | 15 |
| **Total** | **24** | **63** | **55** | **78** | **220** |

Several cells are 0. This is acceptable because the two classifiers are trained independently.

**Train/validation split sizes (DistilBERT):** 176 train / 44 validation, for each of the two classifiers.

**Test set:** the 15 hand-written cases in `eval/eval_runner.py` (Section 11). They are separate from the 220 synthetic rows.

### 4.4 How the train/test split was done

**Actual (synthetic data):** `sklearn.model_selection.train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)`, run separately inside `train()` for each label column.
- Stratifying keeps every class's proportion equal in train and validation.
- It also guarantees that small classes (Sanitation 11, Other 11, Critical 24) appear in validation; in this run 2 of each small category and 5 Critical.
- The split is by **row**. Because every synthetic row is an independent generated sentence, there is no shared source object between rows. However, near-paraphrases from the same generator are possible, and they could inflate validation accuracy.
- The **15 eval cases** were written by hand, separately from the generator, so the Phase 8 numbers come from data the model never saw during training.

**Planned (real data, DATASET_PLAN §6, §10):** split **by photo group**. All 4 complaint variants generated from one photo go to the same split: about 20% of photos → test, stratified by category and severity. Text-only rows are split by row.

- **Why this prevents leakage:** the 4 variants of one photo share the same scene, object, location and often wording, e.g. "light outside main gate not working since 3 days" and "The streetlight outside the main gate hasn't worked since Monday…".
- If variants of the same photo were split across train and test, the model would be tested on near-copies of training sentences, and the multimodal eval would test on photos it had already seen. Accuracy would then measure memorisation of that scene, not generalisation.
- The planned `tools/finalize_dataset.py` validation includes "no `image_id` appears in both splits".

### 4.5 Label quality — who labelled what, and what human verification was done

**Actual:**

| Data | Labelled by | Human verification |
|---|---|---|
| 220 synthetic training rows, `category` | Gemini (generator prompt, T=0.9) | **None.** Filtered only for valid label strings |
| 220 synthetic rows, `severity` | Gemini re-label with a fixed rubric (T=0.0) | **None** row by row. The rubric was written by hand, and the inconsistency was diagnosed by hand (e.g. the chair example) |
| 220 synthetic rows, `has_contradiction` | Gemini (generator) | None; not used for training |
| 15 eval cases | Written and labelled by hand by the developer in `eval/eval_runner.py` (comment: "Hand-label these 15 test cases") | Single annotator. No inter-annotator agreement |

**Planned (real data):**
- Gemini *proposes* each photo's severity and a category sanity check (`tools/build_image_manifest.py`).
- The category owner confirms or edits it in `review/images_<ID>.csv` and sets `verified=yes`. The pipeline refuses to continue while any row is unverified.
- Text rows are reviewed in `review/texts_<ID>.csv`: every `split=test` row plus a random 20% of training rows, with `label_source=human_verified`.
- F2 relabels all test rows from M1/M2/F1 **blind** in `review/test_crosscheck.csv`; F2's own test rows go to M1. **Cohen's κ** is reported, and disagreements are settled by discussion.
- A 10-minute calibration on 8 shared photos comes before any labelling.

### 4.6 The 4 complaint variants per image (planned, DATASET_PLAN §6)

One Gemini call per verified photo returns 4 English variants. Category and severity are **inherited from the verified photo label**, not invented.

| Variant | Description | `has_contradiction` | Example (dead streetlight photo; verbatim from the plan) |
|---|---|---|---|
| `consistent_informal` | Short, casual English: abbreviations, lowercase, small typos | false | "light outside main gate not working since 3 days, v dark at night" |
| `consistent_detailed` | Fuller, well-written English; adds plausible location/duration | false | "The streetlight outside the main gate hasn't worked since Monday and the road is completely dark after 7 pm." |
| `vague` | Minimal information | false | "light not working" |
| `contradicting` | A claim the photo **contradicts** (exaggeration, wrong object, wrong scale) | **true** | "all the lights on campus are sparking and on fire" |

Prompt rules: English only (Indian-English phrasing like "since 3 days" is allowed, Hindi/Hinglish words are not). The contradicting variant must *conflict* with what is visible, not just add unverifiable detail. Length varies from 3 to 40 words. Text-only rows (Internet / Network, Transport, some Security / Other) are generated from a category + severity seed with an empty `image_id`.

### 4.7 Why English only (not Hinglish) for the real dataset

User decision recorded 2026-10-26:
1. **Labels must be human-verifiable by the whole team.** A single language keeps the category and severity review and the blind cross-check consistent.
2. **Whisper and WER.** All recordings are English, so WER is computed against one reference script. The plan can also compare Whisper's English-only `base.en` or force `language="en"`, a cheap accuracy gain. Code-mixed Hinglish has no standard spelling, so WER references would be ambiguous.
3. **Fair Gemini vs DistilBERT comparison.** `distilbert-base-uncased` is an English model. Romanised Hindi tokens split into sub-word fragments that carry little meaning for it, so a Hinglish test set would measure tokenisation failure rather than classification ability.
4. **Scope control before the demo.** The synthetic set, the demo samples and the eval cases were all switched to English. Eval case 1 was changed from the plan's Hinglish text to "light not working outside hostel since 3 days".

---

## SECTION 5 — CUSTOM MODEL 1: DistilBERT CLASSIFIER

### 5.1 What DistilBERT is
DistilBERT is a Transformer **encoder** produced by knowledge distillation from BERT-base. A 6-layer student is trained to match the 12-layer BERT teacher's output distribution, alongside masked-language-modelling and cosine embedding losses. It keeps BERT's hidden size of 768, 12 attention heads and 3072-dim feed-forward layers, but halves the depth and drops the token-type embeddings and pooler. That gives about 66 M parameters, about 40% smaller and about 60% faster than BERT-base, while retaining about 97% of its GLUE performance (Sanh et al., 2019). For classification, `DistilBertForSequenceClassification` takes the `[CLS]` hidden state and passes it through `pre_classifier` (768→768) + ReLU + dropout (`seq_classif_dropout` 0.2) and a final linear layer to `num_labels` logits.

Actual saved config (`models/saved/classifier/*/config.json`): `dim` 768, `n_layers` 6, `n_heads` 12, `hidden_dim` 3072, `vocab_size` 30522, `max_position_embeddings` 512, `dropout` 0.1, `seq_classif_dropout` 0.2, architecture `DistilBertForSequenceClassification`. Parameter count of the category model: **66,961,931**.

### 5.2 What it classifies
There are two **separate** models with identical setups:
- **Category:** 11 classes (`models/saved/classifier/category/`).
- **Severity:** 4 classes (`models/saved/classifier/severity/`).

### 5.3 Exact training setup (actual values; some differ from the requested list)

| Setting | Value | Source |
|---|---|---|
| Base model | `distilbert-base-uncased` | `MODEL_NAME` |
| Tokenizer | `DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")`, `truncation=True, padding=True, max_length=128` | `train()` |
| Padding detail | `padding=True` pads to the **longest sequence in each split**, because each split is tokenized in one call. Inputs are 8–21 words, well under 128 tokens | |
| Train/val split | 80/20, `stratify=labels`, `random_state=42` → 176 / 44 | `train()` |
| **Epochs** | **15** (CLI default `--epochs 15`). The plan's 4 epochs were the first run (val category 0.59, severity 0.48, Critical recall 0) and were replaced | `__main__`, `training_args.bin` |
| Batch size | 16 train / 16 eval → 11 optimisation steps per epoch, 165 steps in total | `TrainingArguments` |
| Optimizer | `adamw_torch_fused` (Trainer default in transformers 5.20.2): β1 0.9, β2 0.999, ε 1e-8, weight decay 0.0, max grad norm 1.0 | read from `training_args.bin` |
| Learning rate | 5e-5 (Trainer default), linear decay, 0 warm-up steps | `training_args.bin` |
| Seed | 42 | `TrainingArguments(seed=42)` |
| Evaluation / saving | `eval_strategy="epoch"`, `save_strategy="epoch"`, `save_total_limit=1` (each checkpoint ≈766 MB) | |
| Model selection | `load_best_model_at_end=True`, `metric_for_best_model="accuracy"`. Accuracy is computed by `evaluate.load("accuracy")` on argmax logits | `compute_metrics` |
| Logging | `logging_steps=10`, `report_to="none"`. `logging_dir` was removed because it no longer exists in transformers 5 | |
| Precision / device | fp32 (`fp16=False`, `bf16=False`), CPU (torch 2.14.1 CPU build) | `training_args.bin` |
| Best epoch | Severity: epoch 10 of 15 (training log in CLAUDE.md). Category: not recorded | |
| Training order | The category model was trained **before** the severity re-label; category labels were unchanged by it. The severity model was trained **after** | CLAUDE.md |

**Validation results (recomputed on 2026-10-27 from the saved models on the same 44-row stratified split):**

Category, **accuracy 0.7955** (35/44); weighted F1 0.80; macro F1 0.78.

| Class | P | R | F1 | n |
|---|---|---|---|---|
| Cleaning / Housekeeping | 1.00 | 0.75 | 0.86 | 4 |
| Electrical | 1.00 | 1.00 | 1.00 | 5 |
| Furniture / Facilities | 0.75 | 0.60 | 0.67 | 5 |
| Infrastructure | 1.00 | 0.60 | 0.75 | 5 |
| Internet / Network | 1.00 | 1.00 | 1.00 | 4 |
| Other | 0.33 | 0.50 | 0.40 | 2 |
| Plumbing / Water | 0.71 | 0.83 | 0.77 | 6 |
| Sanitation | 0.33 | 0.50 | 0.40 | 2 |
| Security | 0.67 | 0.80 | 0.73 | 5 |
| Transport | 1.00 | 1.00 | 1.00 | 3 |
| Waste Management | 1.00 | 1.00 | 1.00 | 3 |

Severity, **accuracy 0.7273** (32/44); weighted F1 0.72; macro F1 0.68.

| Class | P | R | F1 | n |
|---|---|---|---|---|
| Critical | 0.50 | 0.40 | 0.44 | 5 |
| High | 0.73 | 0.92 | 0.81 | 12 |
| Low | 0.88 | 0.64 | 0.74 | 11 |
| Medium | 0.71 | 0.75 | 0.73 | 16 |

Targets set before retraining (CLAUDE.md): category ≥ 0.80, severity ≥ 0.65 with Critical recall > 0.
- Severity met its target (0.727, Critical recall 0.40).
- Category reached 0.7955 — this rounds to the 0.80 target but is just below it.
- Because the best epoch was chosen on this same validation split, these numbers are **optimistically biased**. The independent number is the 15-case eval (Section 11).

**Out-of-distribution probe** (Phase 7; 20 hand-written demo-style sentences; from the CLAUDE.md log): category 16/20, severity 13/20. Known misses:
- street light → Security; pothole → Security
- overflowing toilet → Cleaning; ceiling fan → Infrastructure
- "live wire, sparks" → **Low**: too few Critical hazards in the training data.

### 5.4 Where the models are saved
- `models/saved/classifier/category/` and `models/saved/classifier/severity/`, each holding `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` and `training_args.bin`; 257 MB per directory. Intermediate `checkpoint-*` directories were deleted, so the final models are the files at the directory root.
- `models/saved/` is git-ignored.

### 5.5 Label encoding
- `sklearn.preprocessing.LabelEncoder().fit_transform(df[label_col])` assigns integer ids in **alphabetical** order. The mapping is saved as JSON with **string** keys:
  - `models/saved/labels/category_labels.json` = `{"0": "Cleaning / Housekeeping", "1": "Electrical", "2": "Furniture / Facilities", "3": "Infrastructure", "4": "Internet / Network", "5": "Other", "6": "Plumbing / Water", "7": "Sanitation", "8": "Security", "9": "Transport", "10": "Waste Management"}`
  - `models/saved/labels/severity_labels.json` = `{"0": "Critical", "1": "High", "2": "Low", "3": "Medium"}`
- The model's own `id2label` stays generic (`LABEL_0`…`LABEL_10`).

### 5.6 How inference works
`predict(text, label_col)`:
1. On first call per label, it loads `transformers.pipeline("text-classification", model="models/saved/classifier/<label>", device=-1)` (CPU) and the label JSON into `_loaded_models[label_col]`. If the directory is missing, it returns `None`.
2. It runs `pipe(text, truncation=True, max_length=128)[0]`, which gives `{"label": "LABEL_k", "score": p}` (top-1 softmax).
3. It parses `k = int(label.replace("LABEL_", ""))` and returns `labels[str(k)]`. The plan's code indexed with the integer and raised `KeyError`; this was fixed in Phase 7.

What the model sees:
- **In the app:** the typed text plus the Whisper transcript, joined with a space. For image-only input, it gets Gemini's `issue.problem` summary instead, which is out-of-distribution relative to the raw-complaint training data.
- **In the eval:** the raw case text. If `None` is returned, the eval substitutes `"Other"` / `"Medium"`.
- The score is not used or displayed.

### 5.7 Why two separate models instead of one multi-label/multi-task model
- **Independent axes.** Category ("what is broken") and severity ("how dangerous") are separate decisions. A broken light can be Low (one flickering tube) or High (a dark street at night). With 11 × 4 = 44 joint classes and 220 rows, a single joint softmax would leave many cells with 0–2 examples; 15 of the 44 cells are 0 here.
- **Separate evaluation.** Each axis gets its own accuracy, F1 and confusion, comparable one-to-one with Gemini's `category` and `severity` fields.
- **Independent retraining.** This was exercised in practice: severity labels were re-labelled and the severity model retrained without touching the category model.
- **Trade-off:** two forward passes per complaint and 2 × 257 MB on disk. A shared encoder with two heads would halve both, but would need a custom training loop instead of the stock `Trainer`.

---

## SECTION 6 — CUSTOM MODEL 2: MiniLM CROSS-ENCODER (NLI)

### 6.1 Cross-encoder vs bi-encoder
- **Bi-encoder:** each text is encoded **separately** into a fixed vector (u = f(premise), v = f(hypothesis)), and the pair is scored with a cheap function such as cos(u, v). Vectors can be pre-computed and indexed, which makes it ideal for retrieval. But the two texts never attend to each other, so fine-grained relations such as negation, quantity ("one light" vs "all lights") and scale are lost.
- **Cross-encoder:** the two texts are **concatenated into one input**, e.g. `<s> premise </s></s> hypothesis </s>` for this RoBERTa-type tokenizer. Every self-attention layer attends across both, and a classification head on the first token outputs the relation. It is far more accurate for pairwise judgements, but must be run once per pair.

### 6.2 Model used
`cross-encoder/nli-MiniLM2-L6-H768`, loaded with `sentence_transformers.CrossEncoder(MODEL_NAME)` from the Hugging Face Hub cache (`~/.cache/huggingface/hub/models--cross-encoder--nli-MiniLM2-L6-H768`).
- Config: `model_type` roberta, 6 hidden layers, hidden size 768, **82,120,707 parameters**, `max_length` 512.
- Code path: `models/contradiction_model.py` checks `models/saved/contradiction/` first. That directory does not exist, so the Hub model is always used.

### 6.3 What NLI means
Natural Language Inference: given a **premise** P and a **hypothesis** H, decide whether
- P **entails** H (if P is true, H must be true),
- P **contradicts** H (if P is true, H must be false), or
- the relation is **neutral** (P neither confirms nor rules out H).

The model outputs 3 logits in the order `id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}`, which was verified against the model config in Phase 7.

### 6.4 How it is used here
- **Premise** = image evidence text: `" ".join(issue.observed_evidence)`, the Gemini-extracted list of things visible in the photo. If that list is empty, the premise is `"No image evidence available."`.
- **Hypothesis** = each user claim, taken from `issue.user_reported_facts + issue.inferred_information`.
- The model sees **only the text that Gemini wrote about the image, never the pixels.**
- In the eval runner only, if there is no observed evidence (which is true of all 15 text-only cases), the premise is the **raw case text itself** (`evidence = ... else case["text"]`). That changes the task to "does Gemini's extracted claim contradict the user's own sentence?" (see §11.5).

### 6.5 Label mapping
`status_map = {"entailment": "supported", "neutral": "unverifiable", "contradiction": "contradicted"}`. The outputs then fill the same `ContradictionAudit` lists as the Gemini audit:
- `verified_facts` = supported claims + `observed_evidence`
- `contradicted_claims`, `unverifiable_claims`
- `contradiction_count = len(contradicted)`, `audit_passed = (count == 0)`
- `image_mismatch` is left at its default (False).

### 6.6 Why pre-trained only (no fine-tuning)
- The general NLI task the model was trained on, recognising whether one sentence supports, rules out or says nothing about another, is the same reasoning needed for "does the photo description support this claim?".
- Domain fine-tuning would need labelled **(evidence, claim, relation)** triples. Those are expensive to produce: each needs a photo, a Gemini evidence description, a claim, and a human judgement.
- The synthetic dataset has no such pairs; its `has_contradiction` flag is per complaint, there is no evidence text, and it means "exaggerated", not "contradicted by an image".
- The planned real dataset (4 variants per photo, including a `contradicting` one) will provide such pairs, so `SAVE_DIR = models/saved/contradiction` is kept as the hook for a fine-tuned model.
- The plan's docstring ("Fine-tuning on synthetic data if contradiction pairs are available") describes that future option; no fine-tuning code exists.

### 6.7 Batch inference
`batch_check(claims, evidence_text)`:
1. `pairs = [(evidence_text, claim) for claim in claims]`: N (premise, hypothesis) tuples sharing one premise.
2. `scores_batch = model.predict(pairs)` returns a NumPy array of shape (N, 3). Under sentence-transformers 6.2.0 the default activation for a 3-label head is `Identity`, so these are **raw logits**, not probabilities. For example, ("A broken street lamp is visible.", "The street light is broken.") → [-2.84, 3.04, -0.65] = entailment.
3. For each row: `label_idx = int(scores.argmax())`, `label = label_map[label_idx]`, `score = float(scores.max())` (the winning logit), `status = status_map[label]`.
4. The UI's per-claim reason is `"NLI score: {score:.2f}"`, a logit, not a confidence. No threshold is applied; the decision is a plain argmax.

### 6.8 Comparison to the Gemini contradiction audit

| | Gemini audit (`run_contradiction_audit`) | MiniLM NLI (`run_nli_audit`) |
|---|---|---|
| Input | All claims + all evidence in **one** prompt; can reason across claims and use world knowledge | One (evidence, claim) pair at a time; no cross-claim context, no world knowledge beyond NLI training |
| Output | Per-claim status + a natural-language reason + the whole-input `image_matches_report` / `mismatch_reason` | Per-claim argmax label + logit; no reason, no wrong-photo verdict |
| Determinism | Temperature 0.0, but model fallback means the serving model can change | Fully deterministic, local |
| Cost | 1 API call (quota-limited; ≈2–12 s) | Local CPU, no API, fraction of a second |
| Failure mode observed | With no image, marks every claim "unverifiable" → never "contradicted" (0/3 recall on text-only eval) | Over-flags. A long multi-fact premise often yields "neutral" or "contradiction" for true but more specific claims ("light broken 3 days" vs "streetlight fixture visible" → contradiction) |
| Used for complaint? | **Yes**: its `verified_facts` feed `generate_complaint`, and it triggers the review card | **No**: displayed in the "MiniLM NLI audit (custom)" tab and evaluated offline |

The trade-off is reasoning quality and explanations (Gemini) against cost, independence and reproducibility (MiniLM). The current design trusts Gemini for the user-facing decision and keeps MiniLM as a measurable, explainable comparison.

---

## SECTION 7 — GEMINI PIPELINE (BASELINE)

### 7.1 Model string
- `config.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")`, and `.env` sets `GEMINI_MODEL=gemini-3.5-flash`.
- Every runtime call goes through `_generate`, which tries `[GEMINI_MODEL] + GEMINI_FALLBACK_MODELS` = `gemini-3.5-flash → gemini-3.8-flash → gemini-3.5-flash-lite → gemini-3.1-flash-lite → gemini-3.7-flash`.
- On 2026-10-26/27 `gemini-3.5-flash` had exhausted its daily quota and 3.8-flash returned 503/504, so **all 15 eval cases and all 10 demo-cache entries were served by `gemini-3.5-flash-lite`** (`model_used` in `eval_cases.csv` and in each `demo_cache/*.json`).
- The UI shows the serving model ("Gemini model used: …").
- Retired models: `gemini-2.0-flash` (the plan's default) and `gemini-2.5-flash` return 404.

### 7.2 Stage 1 — Analysis (`analyze_issue`)

**Inputs.** `parts = [combined_text, <image Part>?, "Analyze the above and return the structured JSON as specified."]`:
- `combined_text` is `"USER DESCRIPTION:\n{text}\n\n"` (if text) + `"VOICE TRANSCRIPT:\n{transcript}\n\n"` (if transcript), or `"No text description provided."`.
- The image Part is `image_pil.convert("RGB")` saved as JPEG → `types.Part.from_bytes(data, mime_type="image/jpeg")`. The RGB conversion was added because the sample PNGs are RGBA and JPEG cannot store alpha.

**Config:** `types.GenerateContentConfig(system_instruction=build_analysis_prompt(), temperature=0.1)`.

**Exact system prompt (`ANALYSIS_SYSTEM_PROMPT`; `{categories}` is replaced by the 11 categories joined with ", "; doubled braces render as single braces):**
```
You are CivicFix, a multimodal civic issue analysis assistant.

Your job is to transform unstructured user observations into structured, actionable issue reports.

RULES:
1. Never invent facts not present in the input.
2. Distinguish observed_evidence (from image) from user_reported_facts (from text/voice).
3. List missing_information clearly — do not fill gaps with guesses.
4. Never state uncertain technical conclusions as facts.
5. Assign severity using the provided definitions only.
6. Return ONLY valid JSON matching the schema. No markdown, no explanation outside JSON.
7. If information is ambiguous, state the ambiguity in inferred_information.

CATEGORIES: {categories}

SEVERITY DEFINITIONS:
- Low: Minor inconvenience, no safety impact, few users affected.
- Medium: Meaningful inconvenience, maintenance needed, not urgent.
- High: Significant impact on safety, accessibility, sanitation, or essential services.
- Critical: Immediate danger or serious hazard requiring urgent response.

OUTPUT SCHEMA:
{{
  "problem": "string",
  "category": "one of the categories above",
  "subcategory": "string or null",
  "location": "string or null — only if stated or clearly visible",
  "duration": "string or null — only if stated by user",
  "severity": "Low | Medium | High | Critical",
  "severity_reason": "one sentence explaining severity choice",
  "department": "string",
  "observed_evidence": ["list of facts visible in the image"],
  "user_reported_facts": ["list of facts stated by the user"],
  "inferred_information": ["list of reasonable conclusions with uncertainty noted"],
  "missing_information": ["list of fields that cannot be determined"],
  "recommended_action": "string",
  "confidence": 0.0 to 1.0
}}
```

**Output schema (`src/schemas.py`, verbatim):**
```python
class SeverityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ContradictionFlag(BaseModel):
    claim: str
    status: str  # "supported" | "contradicted" | "unverifiable"
    reason: str


class StructuredIssue(BaseModel):
    problem: str
    category: str
    subcategory: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    severity: SeverityLevel
    severity_reason: str
    department: str
    observed_evidence: List[str] = Field(default_factory=list)
    user_reported_facts: List[str] = Field(default_factory=list)
    inferred_information: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    recommended_action: str
    confidence: float = Field(ge=0.0, le=1.0)


class ContradictionAudit(BaseModel):
    flags: List[ContradictionFlag] = Field(default_factory=list)
    verified_facts: List[str] = Field(default_factory=list)
    contradicted_claims: List[str] = Field(default_factory=list)
    unverifiable_claims: List[str] = Field(default_factory=list)
    contradiction_count: int = 0
    audit_passed: bool = True
    # Photo shows a different problem/scene than the report (e.g. wrong photo uploaded),
    # even when individual claims are only "unverifiable" rather than "contradicted"
    image_mismatch: bool = False
    mismatch_reason: Optional[str] = None


class QualityScore(BaseModel):
    completeness: float = Field(ge=0.0, le=100.0)
    specificity: float = Field(ge=0.0, le=100.0)
    evidence: float = Field(ge=0.0, le=100.0)
    actionability: float = Field(ge=0.0, le=100.0)
    overall: float = Field(ge=0.0, le=100.0)
    raw_input_score: float = Field(ge=0.0, le=100.0)


class CivicFixReport(BaseModel):
    issue: StructuredIssue
    audit: ContradictionAudit
    complaint: str
    quality: QualityScore
    modalities_used: List[str]  # ["text", "image", "audio"]
    gemini_category: str         # Gemini-predicted category (baseline)
    custom_category: Optional[str] = None    # DistilBERT predicted
    custom_severity: Optional[str] = None    # DistilBERT predicted
    unsupported_sentence_count: int = 0
    total_sentence_count: int = 0
```
`category` is a free `str`, not an Enum. Gemini often returns "Main / Sub" strings ("Electrical / Lighting"), which the eval normalises (`normalize_category`) and the UI prefix-matches.

**Retry logic:** `for attempt in range(2)`. Any exception (API error after fallback, JSON decode error, Pydantic `ValidationError`) is logged and retried once; the second failure raises `RuntimeError("Gemini analysis failed after 2 attempts: …")`. The UI turns that into `friendly_error`.

### 7.3 Stage 2 — Contradiction audit (`run_contradiction_audit`)

**How claims are extracted.** Claims are not extracted by a separate call. They are the Stage-1 lists `user_claims = issue.user_reported_facts + issue.inferred_information`.
- If there are no claims, the function returns immediately `ContradictionAudit(flags=[], verified_facts=issue.observed_evidence, audit_passed=True)`, with no API call.
- Otherwise `image_evidence = "\n".join(observed_evidence)` (or `"No image provided."`) and `claims_text = "\n".join(f"- {c}" ...)`.

**Config:** `contents=[prompt]`, `temperature=0.0`, no system instruction.

**Exact prompt (`CONTRADICTION_PROMPT`):**
```
You are a fact-checking assistant for civic complaints.

Compare each USER CLAIM against the IMAGE EVIDENCE and TEXT CONTEXT.
For each claim, determine:
- "supported": the image or context clearly supports this claim
- "contradicted": the image or context contradicts this claim
- "unverifiable": cannot be confirmed or denied from available evidence

USER CLAIMS:
{user_claims}

IMAGE EVIDENCE (what is actually visible):
{image_evidence}

Also decide whether the image shows the SAME problem the user is reporting.
- "image_matches_report": false if the image shows a different problem, object or scene (for example the user
  reports a water leak but the image shows notebook pages or a street light) — i.e. probably the wrong photo.
- true if the image plausibly shows the reported problem, even if some details cannot be confirmed,
  or if no image was provided.

Return ONLY valid JSON:
{{
  "image_matches_report": true,
  "mismatch_reason": "one sentence, only when image_matches_report is false, else null",
  "flags": [
    {{
      "claim": "the user claim",
      "status": "supported | contradicted | unverifiable",
      "reason": "one sentence explanation"
    }}
  ]
}}
```
The prompt mentions "TEXT CONTEXT", but no text-context field is passed. Only claims and image evidence reach the model.

**How flags map to `ContradictionAudit`:**
- `flags = [ContradictionFlag(**f) for f in data["flags"]]`
- `verified_facts = [f.claim for f in flags if f.status == "supported"] + issue.observed_evidence`
- `contradicted_claims = [... status == "contradicted"]`; `unverifiable_claims = [... status == "unverifiable"]`
- `contradiction_count = len(contradicted_claims)`
- `image_mismatch = bool(issue.observed_evidence) and data.get("image_matches_report") is False`. It is set only when there is a photo AND the model explicitly answered false.
- `mismatch_reason = data.get("mismatch_reason") if image_mismatch else None`
- `audit_passed = (contradiction_count == 0) and not image_mismatch`

**Retry / fallback:** 2 attempts. If both fail, it returns `ContradictionAudit(flags=[], verified_facts=issue.observed_evidence + issue.user_reported_facts, audit_passed=True)`. This is **fail-open**: unaudited user facts are treated as verified (limitation, §13).

**Why `image_matches_report` was added (2026-10-27 fix).** A user uploaded a wrong photo (notebook pages) with a water-leak voice note. Every claim came back "unverifiable", so nothing was "contradicted" and no review fired. The whole-input verdict catches a wrong photo even when no individual claim is contradicted.

Live check on `gemini-3.5-flash-lite`:
- 3 matched demo pairs → no review (no false alarms).
- Leak voice + garbage photo, garbage voice + light photo, and the contradiction text + light photo → review, each with a clear reason.

### 7.4 Stage 3 — Complaint generation (`generate_complaint`)

**Inputs:**
- `structured_issue = issue.model_dump_json(indent=2)`: the **entire** StructuredIssue, including `user_reported_facts` and `inferred_information`.
- `verified_facts = "\n".join(f"- {f}" for f in audit.verified_facts)`.

**Config:** `contents=[prompt]`, **temperature 0.3**. Single attempt, no retry (as in the plan). Returns `response.text.strip()`.

**Exact prompt (`COMPLAINT_GENERATION_PROMPT`):**
```
You are generating a formal civic complaint report.

Use ONLY the verified facts provided below. Do not add any information not present in the structured issue.
Do not speculate. Do not use phrases like "it appears" or "possibly" for facts — state verified facts directly.
For genuinely uncertain items, omit them rather than hedging.

Write in formal English. 3-5 sentences. Professional tone.

STRUCTURED ISSUE:
{structured_issue}

VERIFIED FACTS (from contradiction audit):
{verified_facts}

Generate the complaint now. Return ONLY the complaint text, no JSON, no labels.
```

**What "verified facts" means here:** the audit's `verified_facts` = claims Gemini marked **supported** + everything in `observed_evidence` (facts visible in the image).
- Contradicted and unverifiable claims are **not** in this list.
- For a text-only input, no claim can be "supported" (there is no image), so `verified_facts` is empty, and the prompt's "Use ONLY the verified facts" competes with the structured issue JSON. In practice, text-only complaints are written from the structured issue.

**Why verified facts are passed in addition to the raw issue.** The code passes **both**. The structured issue supplies the fields a formal complaint needs (problem, category, location, duration, department, recommended action). The verified-facts list tells the model which *factual assertions* survived the audit. "Use ONLY the verified facts" is therefore enforced by the prompt, not by removing contradicted claims from the JSON.
- This is a design weakness (§13). A stronger variant would remove contradicted claims from the issue JSON, which is exactly what `generate_text_priority_complaint` does for the image fields.

**Text-priority variant (`generate_text_priority_complaint`, temperature 0.3).** It is used when the photo mismatches and the user clicks *Continue with my description*:
- The issue sent is `issue.model_dump(mode="json", exclude={"observed_evidence", "inferred_information"})` of a **re-run** `analyze_issue(text, transcript, None)`, with no image.
- It also receives `user_facts` and `contradicted_claims`.

Exact prompt:
```
You are generating a formal civic complaint report.

The user's photo did NOT match their description (it may be the wrong photo). The user has confirmed their
description is correct, so the description is the trusted source for this complaint.

Rules:
- Base the complaint ONLY on the USER-REPORTED FACTS below. Do not add any other information.
- Ignore image observations; do not describe what the photo shows.
- Do not speculate. Do not use phrases like "it appears" or "possibly".
- End with exactly one sentence stating that the attached photo could not confirm the reported details
  and should be verified on site.

Write in formal English. 3-5 sentences. Professional tone.

STRUCTURED ISSUE (category, severity, location, duration, department):
{structured_issue}

USER-REPORTED FACTS (trusted):
{user_facts}

CLAIMS THE PHOTO CONTRADICTED (keep them as the user's report, but do not present them as photo-verified):
{contradicted_claims}

Generate the complaint now. Return ONLY the complaint text, no JSON, no labels.
```
Why the re-analysis is needed: in a live run, the burst-pipe text plus the light photo was classified "Electrical", because the combined analysis took the problem from the wrong photo. After Continue it became Plumbing / Water with the Water and Sanitation department.

### 7.5 Temperature settings (all Gemini calls)

| Call | Temperature | Rationale |
|---|---|---|
| Analysis | 0.1 | Near-deterministic extraction and classification; a small non-zero value as in the plan |
| Contradiction audit | 0.0 | A verdict should be as repeatable as possible |
| Complaint (both variants) | 0.3 | Slight variation for natural formal prose; still low to limit invention |
| Synthetic data generation | 0.9 | Diversity of wording, typos and topics |
| Severity re-label | 0.0 | Consistency is the whole point of the re-label |

### 7.6 JSON parsing of malformed responses
`_parse_json_response(text)`:
1. `text.strip()`.
2. If it starts with ```` ``` ````, take the content between the first pair of fences (`text.split("```")[1]`), and drop a leading `json` language tag.
3. `json.loads(text.strip())`.

Any `JSONDecodeError` or Pydantic `ValidationError` raised afterwards is caught by the stage's retry loop: 1 retry for analysis and audit, 0 for complaint text, which is not JSON. The schema validation also catches wrong-case severities (Phase 3: "high" is rejected).

Offline tests in Phase 4 (mocked client):
- Malformed JSON was retried once and then succeeded.
- An invalid severity twice → `RuntimeError`.

No `response_mime_type="application/json"` or `response_schema` is used. Structured output relies on the prompt plus validation.

### 7.7 Why two-stage generation (analysis → audit → complaint) instead of one prompt
1. **An audit needs something to audit.** Contradiction checking requires the claims and the evidence to be separated *first* (`user_reported_facts` vs `observed_evidence`). One prompt that extracts, checks and writes at once would hide the intermediate decisions.
2. **Measurability (objective 3).** With separate stages, the same structured issue can produce two complaints, one with the audit's verified facts and one without (every fact passed through), and both can be measured (§9). A single prompt cannot be ablated.
3. **Typed checkpoints.** Each stage output is validated by Pydantic, so a malformed answer is caught at the stage that produced it and retried, instead of corrupting the final text.
4. **Human-in-the-loop.** The pipeline can **stop after the audit** when the photo mismatches, without spending a complaint call, and ask the user. A single prompt would have already written a complaint from a wrong photo.
5. **Swappable components.** The audit can be replaced by MiniLM, and the classifier compared against DistilBERT, because each stage has a defined interface.

Cost: 3 sequential calls, about 25 s per analysis on `gemini-3.5-flash` (Phase 4; ≈8–12 s per call with thinking tokens), and 3× the quota.

---

## SECTION 8 — CONTRADICTION DETECTION

### 8.1 What contradiction detection means in this system
Each **claim** the user makes (text or voice) is compared with **evidence** extracted from the photo, and one of three statuses is assigned:
- **supported**: the image evidence confirms the claim;
- **contradicted**: the image evidence rules the claim out (exaggeration, wrong object, wrong scale);
- **unverifiable**: the image neither confirms nor denies it (e.g. "since 3 days" cannot be seen in a photo).

Separately, a whole-input **image mismatch** verdict ("is this even the same problem?") flags a probably-wrong photo (Gemini only).

Claims = `user_reported_facts + inferred_information`. Gemini's own inferences are also checked, so speculation is audited as well.

Without a photo, no claim can be "contradicted" by evidence. Text-only inputs therefore never trigger the review, by design (`needs_user_review` requires `has_image`).

### 8.2 Two implementations

**A) Gemini-based: `src/ai_service.run_contradiction_audit(issue)`.** One temperature-0.0 call sends all claims and all image evidence together with `CONTRADICTION_PROMPT` (full text in §7.3). It returns per-claim `{claim, status, reason}` plus `image_matches_report` and `mismatch_reason`. It can reason over the whole set and explains each verdict in a sentence. It drives the UI: the review card and the complaint input.

**B) MiniLM NLI: `src/contradiction.run_nli_audit(issue)`.** For each claim, the pair (evidence = `" ".join(observed_evidence)` or "No image evidence available.", claim) goes through `cross-encoder/nli-MiniLM2-L6-H768`. The argmax over {contradiction, entailment, neutral} maps to {contradicted, supported, unverifiable}, and the reason is "NLI score: <logit>". It runs inside `app.run_custom_models`, right after Stage 1, and is shown in the "MiniLM NLI audit (custom)" tab.

### 8.3 How each populates `ContradictionAudit`

| Field | Gemini | MiniLM |
|---|---|---|
| `flags` | `ContradictionFlag(**f)` from the JSON | `ContradictionFlag(claim, status, reason="NLI score: x.xx")` |
| `verified_facts` | supported claims + `observed_evidence` | supported claims + `observed_evidence` |
| `contradicted_claims` | status == contradicted | status == contradicted |
| `unverifiable_claims` | status == unverifiable | status == unverifiable |
| `contradiction_count` | len(contradicted) | len(contradicted) |
| `audit_passed` | no contradictions AND not `image_mismatch` | no contradictions |
| `image_mismatch` / `mismatch_reason` | from `image_matches_report is False` (only if there is observed evidence) | default False / None |
| No claims | `flags=[]`, `verified_facts=observed_evidence`, passed | same |
| Failure | after 2 attempts: `verified_facts = observed_evidence + user_reported_facts`, passed (fail-open) | exception caught in `run_custom_models` → `nli_audit=None`, "MiniLM NLI audit unavailable." |

### 8.4 verified_facts vs contradicted_claims vs unverifiable_claims
- **`verified_facts`**: what the complaint may assert as fact. That is (a) everything Gemini listed as directly **visible** in the photo (`observed_evidence`), plus (b) user or inferred claims the audit marked **supported**.
- **`contradicted_claims`**: user or inferred claims the photo evidence **rules out**. They are excluded from the verified list. If there is a photo, they trigger the review card. In text-priority mode they are passed to the generator explicitly as "keep them as the user's report, but do not present them as photo-verified".
- **`unverifiable_claims`**: claims the photo cannot confirm or deny. They are also excluded from `verified_facts`, but they do not trigger a review. For text-only inputs, **every** claim lands here.

### 8.5 How this feeds into complaint generation
- In `finish_report(result, "verified")`, `generate_complaint(issue, audit)` uses the **Gemini** audit's `verified_facts` as the "VERIFIED FACTS (from contradiction audit)" block. The prompt instructs "Use ONLY the verified facts provided below", and the UI warns "Some claims are contradicted by the image evidence — they were excluded from the complaint."
- The structured issue JSON (which still contains the contradicted claims inside `user_reported_facts`) is also passed (§7.4), so exclusion relies on the model following the instruction.
- If the photo contradicts the text or mismatches it, no verified-mode complaint is generated at all until the user chooses. Choosing Continue switches to the text-priority complaint, which flags the photo.

### 8.6 Why this reduces hallucination — the causal chain
1. LLM complaint writers tend to turn every input statement, and their own guesses, into confident prose. That includes exaggerations ("10-foot craters") and speculation ("faulty wiring").
2. Stage 1 separates *visible* facts from *stated* facts from *inferred* facts, so each assertion has a known source.
3. Stage 2 removes stated and inferred assertions that the image contradicts or cannot confirm from the "verified" list.
4. Stage 3 is told to write only from the verified list and to omit uncertain items rather than hedge.
5. Therefore fewer sentences in the complaint should assert things that are not grounded in the input or the image. That is the unsupported-sentence rate measured in §9.
6. When the photo contradicts the text, the system does not silently pick one: it asks the user. If the user trusts their text, the complaint carries an explicit flag that the photo could not confirm the details.

How strong the evidence is: in the 15-case **text-only** eval, step 3 has almost nothing to remove (no image → no supported claims, no contradictions). The measured effect is therefore small (0.433 vs 0.456). The causal chain is strongest exactly where the eval does not test it yet: inputs with photos.

---

## SECTION 9 — HALLUCINATION REDUCTION

### 9.1 Definition
In this project, a **hallucinated (unsupported) sentence** is a sentence in the generated complaint that is not grounded in any reference fact. Operationally, it shares **fewer than 2 content words** with every fact in the reference list. The **unsupported-sentence rate** = unsupported sentences / total sentences.

### 9.2 The algorithm (`src/scoring.count_unsupported_sentences`; identical logic in `eval/metrics.unsupported_sentence_rate`)
```python
sentences = [s.strip() for s in re.split(r"[.!?]", complaint) if s.strip()]
stopwords = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
             "it", "its", "this", "that", "and", "or", "but", "in", "on", "at",
             "to", "for", "of", "with", "by", "from", "be", "been", "being"}
for sentence in sentences:
    s_words = set(sentence.lower().split()) - stopwords
    supported = any(len(s_words & (set(fact.lower().split()) - stopwords)) >= 2 for fact in verified_facts)
    if not supported: unsupported += 1
return unsupported, len(sentences)          # metrics version also returns rate = unsupported/total (0.0 if total==0)
```
1. **Sentence tokenisation:** a regex split on `.`, `!` or `?`. There is no abbreviation handling, so "e.g." or "3.5" splits incorrectly.
2. **Word tokenisation:** lowercase plus whitespace `split()`. **Punctuation is not stripped**, so "days," ≠ "days", and there is no stemming ("leaking" ≠ "leak").
3. **Stopword removal:** the fixed 29-word set above.
4. **Threshold:** a sentence is supported if **at least one** reference fact shares **≥ 2** remaining words with it (set intersection).

### 9.3 Baseline vs with-audit (as implemented in `eval/eval_runner.py`)
For each of the 15 cases, the *same* `StructuredIssue` produces **two complaints**:
- **With audit:** `generate_complaint(issue, audit)`, where `audit` is the real Gemini audit, so `verified_facts` = supported claims + observed evidence.
- **Without audit (baseline):** `generate_complaint(issue, ContradictionAudit(verified_facts = user_reported_facts + observed_evidence + inferred_information, audit_passed=True))`. Every extracted fact, including inferred and unverifiable ones (and contradicted ones, had there been any), is passed through as "verified".

Both are scored against the **same grounded reference**: `[case["text"]] + issue.user_reported_facts + issue.observed_evidence`, i.e. what the input actually supports. Gemini's own `inferred_information` is deliberately left out of the reference.

**Why not the plan's metric.** The plan scored the *same* complaint against `audit.verified_facts` versus against all facts. Since verified ⊆ all, that metric can never favour the audit. It is kept for transparency under `plan_original_metric_same_complaint`.

### 9.4 Actual results (`eval/results/eval_report.json`)

| Metric | Value |
|---|---|
| `avg_unsupported_rate_with_audit` | **0.4333** |
| `avg_unsupported_rate_without_audit` | **0.4555** |
| Absolute reduction | 0.0222 (2.2 percentage points; 4.9% relative) |
| Plan metric, same complaint vs `audit.verified_facts` | 0.8944 |
| Plan metric, same complaint vs all extracted facts | 0.3444 |

Per case (with → without): 1: 0.33→0.50 · 2: 0.50→0.50 · 3: 0.25→0.50 · 4: 0.50→0.33 · 5: 0.75→0.50 · 6: 0.75→0.67 · 7: 0.50→0.33 · 8: 0.67→0.33 · 9: 0.33→0.33 · 10: 0.50→0.67 · 11: 0.00→0.33 · 12: 0.67→0.75 · 13: 0.50→0.75 · 14: 0.00→0.00 · 15: 0.25→0.33.

The audit is better in 7 cases, worse in 5, and tied in 3. With n = 15 and this per-case spread, the 2.2-point difference is **not statistically meaningful**. The Phase 8 acceptance criterion ("with ≤ without") holds, but by a small margin.

Example (case 1, "light not working outside hostel since 3 days"):
- **Without audit**, the complaint contained "The problem involves either a burnt-out bulb, faulty wiring, or a power supply issue, though the exact cause remains uncertain without an inspection". That sentence is Gemini's own inference written into the complaint.
- **With audit**, the complaint did not contain that sentence.

### 9.5 Limitation of the metric
The 2-word-overlap test is a **lexical proxy**, not semantic grounding:
- **False "supported":** a sentence can share 2 words ("light", "hostel") with a fact while asserting something false.
- **False "unsupported":** a correct paraphrase ("illumination", "three days" vs "3 days") counts as unsupported. Formal complaint boilerplate ("The department is requested to dispatch…") is almost always unsupported because no fact contains those words. This inflates both rates; it explains why even the audited complaints are about 43% "unsupported".
- **No punctuation stripping or stemming** (§9.2).
- A semantic alternative, e.g. NLI entailment of each sentence by the fact set (MiniLM could do this) or human annotation, has not been implemented.

---

## SECTION 10 — QUALITY SCORING

All code is in `src/scoring.py`. It is pure Python with no randomness and no API.

### 10.1 The 4 dimensions — exact formulas

**Completeness** (`score_completeness(issue)`), max 100:
```
+25 if issue.problem
+20 if issue.category
+25 if issue.location
+15 if issue.duration
+15 if issue.department
→ min(score, 100)
```

**Specificity** (`score_specificity(issue)`), max 100:
```
+30 if issue.problem has ≥ 4 words
+20 if issue.subcategory
+25 if issue.severity_reason has ≥ 5 words
+25 if issue.recommended_action has ≥ 5 words
→ min(score, 100)
```

**Evidence** (`score_evidence(issue, audit)`), clamped to [0, 100]:
```
+ min(len(observed_evidence) × 20, 50)
+ min(len(user_reported_facts) × 10, 30)
+20                                  if audit.contradiction_count == 0
− audit.contradiction_count × 10     otherwise
→ max(min(score, 100), 0)
```
(The maximum reachable is 50 + 30 + 20 = 100.)

**Actionability** (`score_actionability(issue)`), max 100:
```
+30 if issue.department
+35 if issue.recommended_action
+20 if issue.severity in ("High", "Critical")   # works because SeverityLevel is a str Enum
+15 if not issue.missing_information
→ min(score, 100)
```
Note: the +20 for High/Critical means a perfectly specified Low-severity issue can score at most 80 on actionability. That is a property of the heuristic.

### 10.2 Overall score
```
overall = round(completeness×0.25 + specificity×0.25 + evidence×0.25 + actionability×0.25, 1)
```
That is an equal-weight mean, 25% each.

### 10.3 Raw input score (`score_raw_input(text, has_image, has_audio)`)
This scores the *unprocessed* input, before any AI:
```
if text:
    + min(word_count / 20, 1.0) × 30                                     # up to 30 for length (20+ words = full)
    + 15 if any of ["block","hostel","outside","near","floor","room","gate","road"] in text.lower()   # location cue
    + 10 if any of ["day","week","hour","since","din","ghante","hafte"] in text.lower()              # duration cue
+ 25 if has_image
+ 20 if has_audio
→ min(round(score, 1), 100)
```
- Matching is by substring, so "today" matches "day" and "roadside" matches "road".
- In the app, `raw_text` is the **typed text only**, not the transcript; voice is credited through `has_audio`.
- The three Hinglish keywords are left over from the original plan.

### 10.4 Why deterministic (no LLM)
- **Reproducible:** the same issue gives the same score every time. Phase 5 verified this over repeated runs.
- **Explainable:** every point traces to a rule a professor can check by hand.
- **Fast and free:** no API call, no quota cost.
- **Independent of the model being graded:** an LLM-as-judge would let Gemini grade Gemini's output.

### 10.5 Before vs after
The UI shows "Before (raw input)" next to "After (AI report)" with a delta, and the Before vs After panel shows "Raw Quality" next to "AI Quality".

Measured values:
- Phase 5 demo cases: text-only 55→83.8, image-only 25→76.2, text+image 77→96.2, contradiction case 62.5→68.8.
- Phase 9 live demo (voice + photo): broken_light 45→96, garbage 45→94, water_leak 45→94.
- 15-case eval (text only): average raw **27.77** → average overall **73.74**.

What it demonstrates: the structuring step adds the fields a maintenance department needs (category, department, severity with a reason, recommended action, explicit missing information). The contradiction case gains least, because contradictions subtract from Evidence.

Caveat: the raw and after scores use **different rubrics** (input cues vs report fields), so the delta shows how much structure was added, not a calibrated quality gain. The after score is largely guaranteed by the schema: problem, category, department and recommended_action are required fields.

---

## SECTION 11 — EVALUATION METHODOLOGY

Run with `python -m eval.eval_runner`. Outputs: `eval/results/eval_report.json`, `eval/results/eval_cases.csv`, and the per-case Gemini cache `eval/results/cache/<sha1(text)[:12]>.json`. The last full run evaluated all 15 cases (`failed_cases: []`), with every case served by `gemini-3.5-flash-lite` and 60 Gemini calls in total (4 per case).

### 11.1 The 15 test cases (verbatim from `TEST_CASES`)

| # | Text | Category | Severity | has_contradiction |
|---|---|---|---|---|
| 1 | light not working outside hostel since 3 days | Electrical | High | False |
| 2 | garbage overflowing near block C since yesterday, smell is unbearable | Waste Management | High | False |
| 3 | water leaking from pipe in boys washroom second floor | Plumbing / Water | Medium | False |
| 4 | chair broken in classroom 204 | Furniture / Facilities | Low | False |
| 5 | wifi not working in hostel for 2 days | Internet / Network | Medium | False |
| 6 | campus bus didnt come today morning | Transport | Medium | False |
| 7 | toilet not clean in ground floor washroom | Sanitation | Medium | False |
| 8 | electricity gone in entire block A since morning | Electrical | Critical | False |
| 9 | road outside college completely destroyed with 10 foot craters | Infrastructure | High | **True** (exaggeration) |
| 10 | drinking water contaminated definitely, everyone is getting sick | Plumbing / Water | Critical | **True** (unverifiable claim) |
| 11 | hostel gate lock broken anyone can enter | Security | High | False |
| 12 | sweeping not done in corridor for a week bad smell | Cleaning / Housekeeping | Medium | False |
| 13 | projector in seminar hall not working | Furniture / Facilities | Medium | False |
| 14 | street light outside gate is broken since 5 days very dangerous | Electrical | High | False |
| 15 | construction debris blocking main road completely impassable for ambulance | Infrastructure | Critical | **True** |

Class supports: categories Electrical 3, Plumbing / Water 2, Infrastructure 2, Furniture / Facilities 2, and 1 each for six others; **Other 0**. Severities: Low 1, Medium 6, High 5, Critical 3. Contradiction positives: 3 of 15. All cases are **text only**: no image and no audio (`image_pil=None`, `transcript=None`, `has_image=False`, `has_audio=False`).

### 11.2 Evaluation axes

**1. Classification accuracy (category and severity, Gemini vs DistilBERT).**
- Metrics: `precision_recall_fscore_support(average="weighted", zero_division=0)` over the full label list, plus the per-class `classification_report`.
- Gemini's category is first normalised: "Main / Sub" → the config category it starts with.
- **Why weighted rather than macro:** the class supports are very unequal (1–3 per class) and "Other" has **zero** support. Macro averaging gives each of the 11 classes equal weight, so the absent class "Other" scores F1 = 0 and drags the macro average down for reasons unrelated to model quality. Gemini's macro F1 is 0.909 although it made **no** errors.
- Weighted averaging weights each class by its support, so the score reflects per-case performance on the distribution actually tested.
- Accuracy is reported alongside. With every label in the label list, weighted recall equals accuracy.

**2. Contradiction detection (Gemini audit vs MiniLM NLI).**
- Binary per case: ground truth `"contradicted"` if `has_contradiction` else `"supported"`.
- Gemini predicts contradicted if `audit.contradiction_count > 0` (`image_mismatch` is not counted).
- MiniLM predicts contradicted if any claim gets the NLI label contradiction.
- Precision, recall and F1 are computed for the positive class "contradicted".
- **Why both matter:**
  - Low **precision** means honest users are falsely accused and pushed into the review step, which costs trust and time.
  - Low **recall** means exaggerated or false claims slip into an official complaint and waste maintenance resources. In the worst case, a Critical dispatch goes out for a non-existent hazard.
  - A detector that flags everything has perfect recall and useless precision, and vice versa, so both are reported, plus F1.

**3. Hallucination reduction.** The unsupported-sentence rate of the with-audit vs without-audit complaint against the same grounded reference (§9.3).

**4. Quality score.** The average `overall` from `calculate_quality_score(issue, audit, case_text, False, False)` across the evaluated cases.

### 11.3 Modality ablation — what it should test, and its status
**Purpose:** measure how much each input modality contributes, by running the same complaint with text only, image only, text + image, text + audio, and all modalities, and comparing accuracy and quality.

The 5 sets defined in `eval/eval_runner.py` (verbatim):
```python
MODALITY_SETS = [
    {"name": "text_only", "use_image": False, "use_audio": False},
    {"name": "image_only", "use_image": True, "use_audio": False},
    {"name": "text_image", "use_image": True, "use_audio": False},
    {"name": "text_audio", "use_image": False, "use_audio": True},
    {"name": "all_modalities", "use_image": True, "use_audio": True},
]
```
**Status: not executed.** `MODALITY_SETS` is defined but never referenced (as in the plan), `results["modality_ablation"]` stays `{}` and is not written to the report, and the 15 cases have no images or audio.
- The sets also have no `use_text` flag, so `image_only` and `text_image` are identical as written.
- A real ablation needs `data/real/test.csv` with `image_file` and `audio_files` columns (DATASET_PLAN §10–11). **Result: [PENDING — requires the real dataset and an `eval_runner.py` extension].**

### 11.4 Actual results (`eval/results/eval_report.json`, every number)

`total_cases`: 15 · `evaluated_cases`: 15 · `failed_cases`: [] · `avg_quality_score`: **73.74**

**Category classification**

| | weighted P | weighted R (= accuracy) | weighted F1 | macro P | macro R | macro F1 |
|---|---|---|---|---|---|---|
| Gemini | 1.0 | 1.0 | **1.0** | 0.9091 | 0.9091 | 0.9091 |
| DistilBERT | 0.5833 | 0.5333 | **0.4822** | 0.4773 | 0.5303 | 0.4455 |

Per class:

| Class (support) | Gemini P / R / F1 | DistilBERT P / R / F1 |
|---|---|---|
| Electrical (3) | 1 / 1 / 1 | 1.0 / 0.333 / 0.50 |
| Plumbing / Water (2) | 1 / 1 / 1 | 0.5 / 1.0 / 0.667 |
| Sanitation (1) | 1 / 1 / 1 | 0 / 0 / 0 |
| Waste Management (1) | 1 / 1 / 1 | 1 / 1 / 1 |
| Infrastructure (2) | 1 / 1 / 1 | 0 / 0 / 0 |
| Security (1) | 1 / 1 / 1 | 0.25 / 1.0 / 0.40 |
| Cleaning / Housekeeping (1) | 1 / 1 / 1 | 0 / 0 / 0 |
| Furniture / Facilities (2) | 1 / 1 / 1 | 1.0 / 0.5 / 0.667 |
| Internet / Network (1) | 1 / 1 / 1 | 0.5 / 1.0 / 0.667 |
| Transport (1) | 1 / 1 / 1 | 1 / 1 / 1 |
| Other (0) | 0 / 0 / 0 | 0 / 0 / 0 |

**Severity classification**

| | weighted P | weighted R (= accuracy) | weighted F1 | macro P | macro R | macro F1 |
|---|---|---|---|---|---|---|
| Gemini | 0.7222 | 0.7333 | **0.7095** | 0.7708 | 0.7667 | 0.7560 |
| DistilBERT | 0.5067 | 0.6667 | **0.5667** | 0.35 | 0.45 | 0.3875 |

| Class (support) | Gemini P / R / F1 | DistilBERT P / R / F1 |
|---|---|---|
| Low (1) | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 |
| Medium (6) | 0.75 / 1.0 / 0.857 | 0.6 / 1.0 / 0.75 |
| High (5) | 0.667 / 0.4 / 0.5 | 0.8 / 0.8 / 0.8 |
| Critical (3) | 0.667 / 0.667 / 0.667 | 0 / 0 / 0 |

**Contradiction detection (positive class = contradicted; 3 positives)**

| | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Gemini audit | 0.0 | 0.0 | 0.0 | 0 | 0 | 3 |
| MiniLM NLI | 0.5 | 0.3333 | 0.4 | 1 | 1 | 2 |

**Hallucination reduction**

| | Value |
|---|---|
| avg unsupported rate, with audit | 0.4333 |
| avg unsupported rate, without audit | 0.4555 |
| plan metric: same complaint vs audit-verified facts | 0.8944 |
| plan metric: same complaint vs all extracted facts | 0.3444 |
| method | "two complaints (with / without audit stage) scored against the same grounded reference: raw input text + user_reported_facts + observed_evidence" |

**Per-case predictions (`eval/results/eval_cases.csv`)**

| # | Gemini cat (raw) | DistilBERT cat | True sev | Gemini sev | DistilBERT sev | Gemini contr. | MiniLM contr. | Raw Q | Overall Q |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Electrical | **Other** ✗ | High | **Medium** ✗ | **Medium** ✗ | supported | supported | 37.0 | 76.2 |
| 2 | Waste Management | Waste Management | High | **Medium** ✗ | High | supported | supported | 40.0 | 78.8 |
| 3 | Plumbing / Water | Plumbing / Water | Medium | Medium | Medium | supported | supported | 28.5 | 72.5 |
| 4 | Furniture / Facilities | Furniture / Facilities | Low | Low | **Medium** ✗ | supported | supported | 22.5 | 72.5 |
| 5 | Internet / Network | Internet / Network | Medium | Medium | Medium | supported | supported | 37.0 | 76.2 |
| 6 | Transport | Transport | Medium | Medium | Medium | supported | supported | 19.0 | 67.5 |
| 7 | Sanitation | **Plumbing / Water** ✗ | Medium | Medium | Medium | supported | supported | 25.5 | 72.5 |
| 8 | Electrical | Electrical | Critical | **High** ✗ | **Medium** ✗ | supported | supported | 37.0 | 81.2 |
| 9 | Infrastructure | **Security** ✗ | High | **Critical** ✗ | High | supported ✗ | supported ✗ | 28.5 | 77.5 |
| 10 | Plumbing / Water | Plumbing / Water | Critical | Critical | **Medium** ✗ | supported ✗ | supported ✗ | 12.0 | 71.2 |
| 11 | Security | Security | High | High | High | supported | supported | 25.5 | 71.2 |
| 12 | Cleaning / Housekeeping | **Plumbing / Water** ✗ | Medium | Medium | Medium | supported | **contradicted** (FP) | 25.0 | 70.0 |
| 13 | Furniture / Facilities | **Internet / Network** ✗ | Medium | Medium | Medium | supported | supported | 9.0 | 63.8 |
| 14 | Electrical | **Security** ✗ | High | High | High | supported | supported | 41.5 | 83.8 |
| 15 | Infrastructure | **Security** ✗ | Critical | Critical | **High** ✗ | supported ✗ | **contradicted** (TP) | 28.5 | 71.2 |

(In this run every Gemini category was returned as an exact config category, so normalisation changed nothing.)

### 11.5 Interpretation

- **Category: Gemini wins clearly (100% vs 53%).**
  - Gemini is a very large model with broad world knowledge about which department handles which problem.
  - DistilBERT has seen 176 short synthetic sentences from the same generator, and its errors follow the training data's biases. It over-predicts **Security**, the joint-largest training class (27 rows, many about "dark/unsafe" areas), for a street light (case 14), a destroyed road (case 9) and debris (case 15). It maps "toilet" and "sweeping…smell" to Plumbing, and "projector" to Internet / Network.
  - Its validation accuracy (0.80) vs eval accuracy (0.53) shows a **distribution gap** between the synthetic data and hand-written test sentences.
- **Severity: Gemini wins narrowly (73% vs 67%; weighted F1 0.71 vs 0.57).**
  - DistilBERT **never predicted Low or Critical** on the test set: every prediction was Medium or High, the two largest training classes (78 and 63 rows). That gives it a high recall on Medium (1.0) and good High F1 (0.80), but 0 on the extremes, hence macro F1 0.39.
  - Gemini's errors are both up and down: High→Medium ×2, Critical→High, High→Critical. Some of these are arguably label ambiguity. Case 9 is labelled High by the annotator, but its text claims "10 foot craters", which by the rubric would be Critical.
- **Contradiction: neither works on text-only input, as expected by construction.**
  - With no image, the Gemini audit has no evidence to contradict against and marks claims "unverifiable", never "contradicted": recall 0/3.
  - MiniLM's one true positive (case 15) and one false positive (case 12) come from comparing Gemini-extracted claims against the **raw case text as premise** (the eval-only fallback). That is a different task from claim-vs-image, so these numbers say little about MiniLM's value for photo evidence.
  - All 3 ground-truth positives are **exaggerations or unverifiable claims within the text**, which no image-evidence auditor can catch without an image.
  - Conclusion: this axis needs photo-claim pairs; it is the main motivation for the real dataset's `contradicting` variant.
- **Hallucination: a small reduction in the expected direction** (0.433 vs 0.456). Text-only inputs give the audit almost nothing to remove, so a small effect is expected; the main visible difference is that the no-audit complaints carry Gemini's speculative inferences (case 1).
- **Quality:** average 73.74 after vs 27.77 raw. Structuring consistently adds department, action, severity reasoning and explicit missing information.
- **Overall:** for the user-facing pipeline, Gemini is the stronger component on every measured axis except contradiction, where both fail on text-only data. The custom models are faster, free and deterministic, but limited by synthetic training data. This supports the design choice of Gemini for decisions and the custom models for comparison and research.
- **Phase 10 note:** the Phase 10 criterion "eval in under 5 minutes" is met only with the cache. The first uncached run took about 18 minutes because of fallback waits.

---

## SECTION 12 — DESIGN DECISIONS AND TRADEOFFS

**1. Synthetic data first, real data later**
- **Decision:** train DistilBERT on 220 Gemini-generated synthetic complaints now; collect the real photo/voice dataset after the demo.
- **Why:** the demo date (2026-10-28) left no time to collect, label and review about 650 real texts across 4 people. Synthetic data made the custom-model pipeline (training, inference, eval, UI comparison) work end to end on day 1. The plan's Real Data Swap Guide keeps the CSV format compatible.
- **Rejected:** waiting for the real data before training (no demo); hand-writing training data (too slow, and inconsistent across writers).
- **Tradeoff accepted:**
  - Labels are Gemini-generated, with no human verification. Severity labels were inconsistent until re-labelled.
  - Real-world validity is low (val 0.80 vs eval 0.53 category accuracy).
  - Using Gemini to make the training labels and also as the baseline adds some circularity, partly offset by hand-labelled test cases.

**2. Two-stage generation (analysis → audit → complaint)**
- **Decision:** separate calls for structured extraction, audit and complaint writing.
- **Why:** it makes the audit possible and measurable (§7.7). Each stage is Pydantic-validated, and the pipeline can stop for user review before writing.
- **Rejected:** a single "analyse and write a complaint" prompt.
- **Tradeoff accepted:** 3× the API calls (3 per analysis against a 20/day/model quota) and about 25 s latency, mitigated by the fallback chain and caches.

**3. DistilBERT over full BERT or GPT**
- **Decision:** `distilbert-base-uncased`, fine-tuned twice (category and severity), on CPU.
- **Why:** it trains in minutes on CPU with 176 rows and gives fast, free, deterministic local inference. It is a genuine custom-trained model for objective 1.
- **Rejected:**
  - BERT-base or RoBERTa: about 2× the compute, and little expected gain at this data size.
  - A GPT or Gemini zero-shot classifier: that is the baseline itself, not a custom model.
- **Tradeoff accepted:** lower accuracy than Gemini on hand-written cases; it is sensitive to training-data bias.

**4. MiniLM NLI over rule-based contradiction**
- **Decision:** a pre-trained `cross-encoder/nli-MiniLM2-L6-H768` as the custom contradiction checker.
- **Why:** it is purpose-built for entailment and contradiction, catches paraphrase and scale conflicts that keyword rules cannot, needs no training data, runs locally, and is independent of Gemini.
- **Rejected:** keyword/negation rules (brittle); a second LLM checker (cost, quota, opacity); fine-tuning (no claim-evidence pairs).
- **Tradeoff accepted:** it sees only Gemini's text description of the image, not the image; it over-flags; it produces uncalibrated logits; and it is shown but not used for the user-facing decision.

**5. Deterministic quality score over an LLM-graded score**
- **Decision:** a rule-based 4-dimension score plus a raw-input score.
- **Why:** reproducible, explainable rule by rule, instant, free, and not "Gemini grading Gemini".
- **Rejected:** an LLM-as-judge rubric score.
- **Tradeoff accepted:** the heuristics are not benchmarked against human judgements. The score rewards the presence of fields rather than their correctness, and it favours High/Critical severities in Actionability.

**6. Pydantic validation with retry**
- **Decision:** validate every Gemini JSON into typed models; retry analysis and audit once on any parse or validation error.
- **Why:** malformed or out-of-schema LLM output (wrong-case severity, confidence > 1, Markdown-fenced JSON) must not reach the UI or the scorer. One retry fixes most transient format errors.
- **Rejected:** unvalidated `dict` access (crashes later and invisibly); unlimited retries (quota).
- **Tradeoff accepted:**
  - A failure costs one extra call.
  - After 2 failures, analysis raises, but the audit **fails open** (treats user facts as verified).
  - `ContradictionFlag.status` and `category` are not enums, so some invalid values pass validation.

**7. Gemini temperature 0.1 for analysis and 0.3 for complaint generation** (plus 0.0 for the audit and the re-label, and 0.9 for generation)
- **Decision:** low temperature where output is a decision, slightly higher where output is prose.
- **Why:** extraction and classification should be nearly repeatable; the audit is fully greedy (0.0). The complaint benefits from slightly varied, natural formal wording, but higher values increase invention. Data generation needs diversity.
- **Rejected:** a single temperature for all calls; temperature 0 for the complaint (repetitive, stilted prose was the expected effect; this was not tested).
- **Tradeoff accepted:** no output is truly reproducible, because the serving model can change via fallback, and the thinking models are not bit-deterministic even at 0.0. The caches provide reproducibility for the demo and the eval instead.

**8. Photo-grouped train/test split** (planned for the real dataset; **not yet implemented**)
- **Decision:** all texts generated from one photo go to the same split; about 20% of photos → test, stratified by category and severity; text-only rows split by row.
- **Why:** prevents leakage between the near-duplicate variants of one scene (§4.4). The test set then measures generalisation to unseen photos and scenes, which also matters for the multimodal eval.
- **Rejected:** a random row-level split (inflates accuracy by testing on paraphrases of training rows).
- **Tradeoff accepted:**
  - Fewer, lumpier test groups make exact stratification harder; small categories (Transport, Internet / Network, Other) may get only 1–2 test photos.
  - The current synthetic split is still row-level.

---

## SECTION 13 — LIMITATIONS

The limitations you listed:
1. **Severity is AI-estimated, not professionally validated.**
   - Gemini assigns it from a 4-line rubric.
   - DistilBERT learned it from Gemini-relabelled synthetic labels.
   - The 15 test labels come from a single annotator.
   - No facilities professional reviewed any severity.
2. **The 2-word-overlap unsupported-sentence rate is a weak proxy.** It is lexical, with no punctuation stripping, no stemming and no paraphrase handling, and it counts formal boilerplate as unsupported (§9.5). The measured with/without-audit difference (0.022) is within noise at n = 15.
3. **Synthetic training data may not match the real complaint distribution.**
   - 220 rows, 8–21 words, from 2 Gemini models.
   - Validation 0.80 vs eval 0.53 category accuracy shows the gap.
   - Labels are not human-verified.
4. **MiniLM NLI works on text evidence only.**
   - Its premise is Gemini's description of the image, so any Gemini perception error propagates.
   - It never sees pixels.
   - In the eval, the premise falls back to the case text itself.
5. **15 test cases is a small evaluation set.**
   - 1–3 examples per class, "Other" absent, 3 contradiction positives.
   - No confidence intervals.
   - One annotator.
   - Text only.
6. **Whisper `base` may underperform on accented English.**
   - Only Windows TTS clips and one TTS speech file were tested.
   - No WER has been measured.
   - The multilingual `base` model with auto-detection is used, not `base.en` / `language="en"`.
7. **No duplicate detection across complaints.** There is no complaint store, so repeated reports of the same issue are not merged.
8. **Quality score dimensions are heuristic, not benchmarked.** Weights (25% each) and point values were set by design, not fitted to human ratings. The score rewards field presence, not correctness.

Additional limitations found while preparing this report:

9. **The complaint generator receives the full structured issue as well as the verified facts.** Exclusion of contradicted or unverifiable claims depends on the model obeying the prompt; it is not enforced in code (§7.4).
10. **The audit fails open.** If both audit attempts fail, `verified_facts = observed_evidence + user_reported_facts` and `audit_passed=True`, so unaudited claims are treated as verified without telling the user.
11. **Contradiction eval is structurally weak.** The test set has no images, so the Gemini audit cannot return "contradicted" (recall 0 by construction). The ground-truth "contradictions" are text exaggerations, not photo conflicts.
12. **The modality ablation has not been run**, so objective 4's "modality ablation" is unmet. `MODALITY_SETS` is unused and lacks a `use_text` flag.
13. **The MiniLM audit does not affect the output.** It is shown for comparison only, but the sidebar architecture diagram implies it feeds the complaint.
14. **The MiniLM "score" is a raw logit,** displayed as "NLI score", not a calibrated probability. No decision threshold is applied (argmax only).
15. **The serving model is not fixed.** Fallback means results may come from different Gemini variants. Eval and demo results came from `gemini-3.5-flash-lite`, not the configured `gemini-3.5-flash`.
16. **Free-tier quota of 20 requests/day/model.** A full uncached eval (60 calls) needs several models' quotas. The caches can serve **stale** results if prompts change; they must be deleted manually (`demo_cache/`, `eval/results/cache/`).
17. **Validation numbers are optimistic.** The best epoch was selected on the same 44-row validation split that reports the accuracy.
18. **Severity imbalance in the custom model.** DistilBERT predicted no Low or Critical on the test set; Critical has only 24 training rows, and one hazard probe ("live wire, sparks") was classified Low.
19. **English only.** The scorer still has 3 Hinglish keywords, and the app does not support Hindi or Hinglish input as a requirement.
20. **Gemini `category` and `department` are free strings.** They do not always match `config.CATEGORIES` / `config.DEPARTMENTS` ("Main / Sub" categories), so the UI agreement check uses prefix matching.
21. **Image privacy.** Photos are sent to Gemini. Free-tier data may be used by Google to improve its products (DATASET_PLAN §4.2), and the app does not blur faces or plates.
22. **Phase 10 (the final manual test matrix of 15 scenarios) has not been run.** Robustness claims rest on the per-phase tests logged in CLAUDE.md.
23. **Repository hygiene.**
    - `.env.example` is **missing** from the working tree and from git, although `README.md` says `cp .env.example .env` and CLAUDE.md requires it to exist and stay blank.
    - `.claude/settings.local.json` is tracked in git.
    - `data_generator.py` overwrites the re-labelled `complaints.csv` if re-run.

---

## SECTION 14 — HOW TO RUN EVERYTHING

Commands are for Windows PowerShell from the project root `C:\Users\khatr\Desktop\rag_project`. Always use the venv interpreter `.\.venv\Scripts\python.exe`.

### 1. Environment setup
```powershell
# Python 3.13.7 was used; plan minimum is 3.10
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check          # should print "No broken requirements found."

# Create .env (never commit it). .env.example is currently missing — create .env directly:
@'
GEMINI_API_KEY=<your key>
GEMINI_MODEL=gemini-3.5-flash
'@ | Out-File -Encoding utf8 .env
# Optional: GEMINI_FALLBACK_MODELS=gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.7-flash
# (do NOT add an empty GEMINI_FALLBACK_MODELS= line — that disables fallback)
# Quota tip: override the model for one session without editing .env (load_dotenv does not override):
$env:GEMINI_MODEL = 'gemini-3.1-flash-lite'

.\.venv\Scripts\python.exe -c "import config; print(config.CATEGORIES)"   # sanity check
```
- No system ffmpeg is needed: `imageio-ffmpeg` provides one.
- The first audio analysis downloads Whisper `base` (≈139 MB) to `~/.cache/whisper`.
- The first custom-model run downloads `distilbert-base-uncased` and `cross-encoder/nli-MiniLM2-L6-H768` to `~/.cache/huggingface/hub`.

### 2. Train custom models
```powershell
# (a) OPTIONAL — regenerate synthetic data. WARNING: overwrites data/synthetic/complaints.csv (the current
#     re-labelled 220-row file). Back it up first. Costs ~10 Gemini calls; calls GEMINI_MODEL directly (no fallback).
Copy-Item data\synthetic\complaints.csv data\synthetic\complaints_backup.csv
.\.venv\Scripts\python.exe -m models.data_generator

# (b) OPTIONAL — re-label severity with the fixed rubric (4 Gemini calls for 220 rows, uses model fallback)
.\.venv\Scripts\python.exe -m models.relabel_severity

# (c) Train both DistilBERT classifiers (category, then severity). Default 15 epochs; CPU; several minutes each.
.\.venv\Scripts\python.exe -m models.classifier --train
#     options: --data <csv>  --epochs <n>
#     outputs: models/saved/classifier/{category,severity}/  and  models/saved/labels/{category,severity}_labels.json

# (d) MiniLM needs no training — it downloads on first use. Quick check:
.\.venv\Scripts\python.exe -c "from models.contradiction_model import check_contradiction as c; print(c('The street is flooded.', 'A broken street lamp is visible.'))"
```

### 3. Run the app
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
# → http://localhost:8501 . The file watcher is disabled (.streamlit/config.toml): restart after code edits.

# Headless smoke test (no browser):
.\.venv\Scripts\python.exe -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('app.py', default_timeout=120).run(); print(at.exception)"
```
**Demo inputs:**
- `sample_data/{broken_light,garbage,water_leak}.wav` + the matching `.png`. These are pre-cached in `demo_cache/`, so they work with 0 quota when the sidebar toggle "Reuse cached results for identical inputs" is on.
- Contradiction demo: type "A pipe burst and the whole street is flooded, water everywhere for a week." and upload `broken_light.png`. This is also cached, including the Continue path.
- Backup transcripts are in `sample_data/voice_samples.md`.
- After any **mocked** AppTest, delete the fake entries it writes to `demo_cache/`.

### 4. Run the evaluation
```powershell
.\.venv\Scripts\python.exe -m eval.eval_runner
# → eval/results/eval_report.json, eval/results/eval_cases.csv (re-uses eval/results/cache/*.json: 0 Gemini calls)

# Force fresh Gemini outputs (60 calls; exceeds one model's daily free quota → relies on fallback):
Remove-Item -Recurse -Force eval\results\cache
.\.venv\Scripts\python.exe -m eval.eval_runner
```
The sidebar "Evaluation Summary" in the app reads `eval/results/eval_report.json` automatically.

### 5. Swapping in real data
1. Run the DATASET_PLAN steps ("Dataset Step 1…7"). Each step's `tools/*.py` script still has to be built; none exists yet. The final export is `data/real/train.csv` and `data/real/test.csv`, with at least the columns `raw_complaint, category, severity, has_contradiction`. Category and severity values must exactly match `config.py`, and every class needs enough rows for the stratified split (≥ 2 per class; the plan targets ≥ 20 per category and ≥ 40 per severity).
2. Retrain on the real data (no code change):
   ```powershell
   .\.venv\Scripts\python.exe -m models.classifier --train --data data/real/train.csv
   ```
   The build plan's Swap Guide names `data/real/complaints.csv`; DATASET_PLAN supersedes it with `train.csv`.
3. Evaluation on `data/real/test.csv` **requires extending `eval/eval_runner.py`**. It currently uses the hard-coded `TEST_CASES`. The extension would load `test.csv`, pass `image_file` / `audio_files` to `analyze_issue` / `transcribe_audio`, implement the `MODALITY_SETS` loop, and compute WER. This is to be decided when that phase is reached.
4. Optionally fine-tune MiniLM on the real (evidence, claim, relation) pairs and save it to `models/saved/contradiction/`. `contradiction_model._get_model()` already loads that directory if it exists.
5. Delete `demo_cache/` and `eval/results/cache/` after any prompt or model change, so stale outputs aren't reused.

---

## SECTION 15 — FILE TREE

Snapshot of the working tree on 2026-10-27, excluding `.venv/`, `.git/` and `__pycache__/`. `git status` at the start of the session showed uncommitted modifications (marked *mod*) and one untracked directory (marked *untracked*).

```
rag_project/
├── .claude/
│   └── settings.local.json          Claude Code local permission settings for this project (tracked in git)
├── .env                             Secrets: GEMINI_API_KEY + GEMINI_MODEL=gemini-3.5-flash (git-ignored; never printed)
├── .gitignore                       Ignores .env, venvs, caches, models/saved/, eval/results/*.json|csv, data CSVs, demo_cache/, eval/results/cache/
├── .streamlit/                      (untracked)
│   └── config.toml                  [server] fileWatcherType = "none" — silences torchvision tracebacks; restart app after edits
├── CIVICFIX_BUILD_PLAN.md           Phase-gated build plan (Phases 0–10, Real Data Swap Guide, Key Design Decisions) — the spec
├── CIVICFIX_TECHNICAL_REPORT.md     This report
├── CLAUDE.md                        (mod) Project memory: workflow rules, environment, quota notes, phase log with every deviation
├── DATASET_PLAN.md                  Real-dataset plan for the 4-member team (photos, 4 text variants, voice, review, split) — deferred
├── README.md                        Setup / train / run / evaluate commands and notes (synthetic data, quota fallback, demo cache)
├── app.py                           (mod) Streamlit UI + pipeline orchestration, review flow, custom-model comparison, caches, report download
├── config.py                        Env loading, GEMINI_MODEL + fallback list, 11 CATEGORIES, 4 SEVERITY_LEVELS, DEPARTMENTS map
├── requirements.txt                 18 dependencies with minimum versions (adds imageio-ffmpeg, accelerate to the plan's list)
│
├── src/
│   ├── __init__.py                  Empty package marker
│   ├── ai_service.py                (mod) Gemini client, model fallback (_generate), JSON parsing, analysis / audit / complaint / text-priority complaint
│   ├── contradiction.py             run_nli_audit: MiniLM NLI audit → ContradictionAudit
│   ├── prompts.py                   (mod) ANALYSIS_SYSTEM_PROMPT, COMPLAINT_GENERATION_PROMPT, TEXT_PRIORITY_COMPLAINT_PROMPT, CONTRADICTION_PROMPT, build_analysis_prompt
│   ├── schemas.py                   (mod) Pydantic v2 models: SeverityLevel, ContradictionFlag, StructuredIssue, ContradictionAudit, QualityScore, CivicFixReport
│   ├── scoring.py                   Deterministic quality score (4 dimensions + raw input) and unsupported-sentence counter
│   ├── speech_service.py            Whisper "base" transcription with bundled-ffmpeg fallback
│   └── utils.py                     save_temp_audio, load_image_pil, image_to_base64 (unused)
│
├── models/
│   ├── __init__.py                  Empty package marker
│   ├── classifier.py                DistilBERT train (Trainer, 15 epochs default) + cached pipeline inference for category/severity
│   ├── contradiction_model.py       MiniLM cross-encoder wrapper: check_contradiction, batch_check
│   ├── data_generator.py            Gemini synthetic complaint generator (English only, T=0.9, 20/batch) → data/synthetic/complaints.csv
│   ├── relabel_severity.py          Re-labels synthetic severity with one rubric via Gemini (T=0.0, 55 rows/call)
│   └── saved/                       (git-ignored) trained artefacts
│       ├── classifier/
│       │   ├── category/
│       │   │   ├── config.json            DistilBertForSequenceClassification config, 11 labels (LABEL_0…LABEL_10)
│       │   │   ├── model.safetensors      Fine-tuned category weights (66,961,931 params; dir 257 MB)
│       │   │   ├── tokenizer.json         Fast tokenizer vocabulary/merges (distilbert-base-uncased)
│       │   │   ├── tokenizer_config.json  Tokenizer settings (lowercase, special tokens)
│       │   │   └── training_args.bin      Pickled TrainingArguments (lr 5e-5, 15 epochs, batch 16, seed 42, adamw_torch_fused)
│       │   └── severity/
│       │       ├── config.json            Same architecture, 4 labels (LABEL_0…LABEL_3)
│       │       ├── model.safetensors      Fine-tuned severity weights (trained after the re-label; dir 257 MB)
│       │       ├── tokenizer.json         Fast tokenizer
│       │       ├── tokenizer_config.json  Tokenizer settings
│       │       └── training_args.bin      Pickled TrainingArguments (same values)
│       └── labels/
│           ├── category_labels.json       {"0": "Cleaning / Housekeeping", …, "10": "Waste Management"} (alphabetical LabelEncoder ids)
│           └── severity_labels.json       {"0": "Critical", "1": "High", "2": "Low", "3": "Medium"}
│
├── eval/
│   ├── __init__.py                  Empty package marker
│   ├── eval_runner.py               15 TEST_CASES, MODALITY_SETS (unused), category normaliser, cached Gemini runs, report writer
│   ├── metrics.py                   Weighted classification metrics, contradiction P/R/F1, unsupported-sentence rate
│   └── results/
│       ├── .gitkeep                 Keeps the folder in git
│       ├── eval_report.json         Aggregate metrics of the last run (Section 11.4)
│       ├── eval_cases.csv           Per-case predictions, scores and both complaints (with/without audit)
│       └── cache/                   (git-ignored) per-case Gemini outputs: issue, audit, complaint, complaint_no_audit, model_used
│           ├── f63285850b03.json    Case 1  — light not working outside hostel since 3 days
│           ├── 189fc097c613.json    Case 2  — garbage overflowing near block C…
│           ├── 47a95d17a80a.json    Case 3  — water leaking from pipe in boys washroom…
│           ├── a42fcd1c7bd7.json    Case 4  — chair broken in classroom 204
│           ├── f40ae4b47892.json    Case 5  — wifi not working in hostel for 2 days
│           ├── 45ef9466da36.json    Case 6  — campus bus didnt come today morning
│           ├── a4a5c2e5019c.json    Case 7  — toilet not clean in ground floor washroom
│           ├── e7038b6cf921.json    Case 8  — electricity gone in entire block A…
│           ├── 0597aeafa426.json    Case 9  — road outside college completely destroyed…
│           ├── fca651e576f8.json    Case 10 — drinking water contaminated definitely…
│           ├── c4241ed6cd8c.json    Case 11 — hostel gate lock broken anyone can enter
│           ├── 064ce10c2b85.json    Case 12 — sweeping not done in corridor…
│           ├── 445e6e616f6e.json    Case 13 — projector in seminar hall not working
│           ├── 0c217325cf23.json    Case 14 — street light outside gate is broken…
│           └── e5fe31de3722.json    Case 15 — construction debris blocking main road…
│
├── data/
│   ├── real/
│   │   └── .gitkeep                 Placeholder — real dataset not collected yet
│   └── synthetic/
│       ├── .gitkeep                 Keeps the folder in git
│       ├── complaints.csv           Training set: 220 rows (raw_complaint, category, severity [re-labelled], has_contradiction, source_model, severity_original)
│       ├── complaints_batch1.csv    Raw generator output part 1: 80 rows from gemini-3.5-flash
│       ├── complaints_batch2.csv    Raw generator output part 2: 140 rows from gemini-3.1-flash-lite
│       └── complaints_before_relabel.csv  Backup of the 220-row set with the original Gemini severities
│
├── demo_cache/                      (git-ignored) Gemini outputs keyed by sha1(text, transcript, image bytes)[:16]; all served by gemini-3.5-flash-lite
│   ├── 0c27806fdd6cef31.json        Garbage voice + mismatched photo → review (2 contradicted, image_mismatch)
│   ├── 38630265289a2891.json        Garbage voice input — issue/audit only (older format, no mismatch field)
│   ├── 39bba160677b48c1.json        Contradiction demo (burst-pipe text + broken_light.png) incl. text-priority re-analysis + complaint
│   ├── 4d105fc823676ac0.json        Water-leak demo (voice + water_leak.png) — matched, complaint cached
│   ├── 4f40c26bfa81b6c0.json        User's own run: broken CCTV camera report — matched, complaint cached
│   ├── 5f124501481eb0c1.json        Broken-light demo (voice + broken_light.png) — matched, complaint cached
│   ├── 7c524701649ecdbf.json        Garbage demo (voice + garbage.png) — matched, complaint cached
│   ├── bb3b654b9090afe0.json        Garbage text + mismatched photo → review (3 contradicted) incl. text-priority path
│   ├── ca37112ee628fc9e.json        User's own run: CCTV report + mismatched photo → review (image_mismatch)
│   └── cf001c3c04dee3db.json        Water-leak voice + garbage photo → review (2 contradicted, image_mismatch)
│
└── sample_data/
    ├── .gitkeep                     Keeps the folder in git
    ├── broken_light.png             Demo photo (user-supplied, RGBA PNG): broken street lamp
    ├── broken_light.wav             Windows-TTS voice: street light outside boys hostel gate not working 3 days
    ├── garbage.png                  Demo photo: overflowing garbage bin
    ├── garbage.wav                  Windows-TTS voice: bin near canteen overflowing since Monday
    ├── water_leak.png               Demo photo: water leak from pipe
    ├── water_leak.wav               Windows-TTS voice: leak on second-floor corridor of academic block, 2 days
    └── voice_samples.md             (mod) Backup transcripts of the 3 voice clips + the contradiction-demo instructions
```

Outside the repository but required at run time:
- `~/.cache/whisper/` (Whisper `base`, 139 MB).
- `~/.cache/huggingface/hub/models--distilbert-base-uncased`, `models--cross-encoder--nli-MiniLM2-L6-H768`. A `models--ProsusAI--finbert` entry is also present; it is unrelated to this project.
- `%TEMP%\civicfix_ffmpeg\ffmpeg.exe` (copied from imageio-ffmpeg on first transcription).

Missing although referenced: `.env.example` (referenced by `README.md` and `CLAUDE.md`; absent from both disk and git).
