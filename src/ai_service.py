import json
import logging
import time
from typing import Optional
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image
import io

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODELS
from src.schemas import StructuredIssue, ContradictionAudit, ContradictionFlag
from src.prompts import build_analysis_prompt, COMPLAINT_GENERATION_PROMPT, CONTRADICTION_PROMPT

logger = logging.getLogger(__name__)

_client = None
_exhausted_models = set()  # models that hit their daily quota (429) or are retired (404) — skipped for this process
_cooldown_until = {}       # model -> time.time() until which it's skipped after an overload (503/504)
_COOLDOWN_SECONDS = 600    # a 504 costs ~60 s, so don't retry an overloaded model on every call
_FALLBACK_CODES = (404, 429, 503, 504)

last_model_used = None  # which model served the most recent request (shown in the UI)


def _get_client():
    global _client
    if _client is None:
        # timeout (ms) so an overloaded model surfaces an error instead of hanging the UI
        _client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=120_000))
    return _client


def _generate(contents, config):
    """generate_content with model fallback: free-tier quota is per model, so on 429/503/504/404 try the next one."""
    global last_model_used
    client = _get_client()
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    now = time.time()
    candidates = [
        m for m in models if m not in _exhausted_models and _cooldown_until.get(m, 0) <= now
    ] or [m for m in models if m not in _exhausted_models] or models
    last_error = None
    for model in candidates:
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            last_model_used = model
            _cooldown_until.pop(model, None)
            return response
        except genai_errors.APIError as e:
            if e.code not in _FALLBACK_CODES:
                raise
            if e.code in (404, 429):
                _exhausted_models.add(model)
            else:
                _cooldown_until[model] = time.time() + _COOLDOWN_SECONDS
            logger.warning(f"{model} failed with {e.code}; trying next model")
            last_error = e
    raise last_error


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def analyze_issue(
    text: Optional[str],
    transcript: Optional[str],
    image_pil: Optional[Image.Image],
) -> StructuredIssue:
    combined_text = ""
    if text:
        combined_text += f"USER DESCRIPTION:\n{text}\n\n"
    if transcript:
        combined_text += f"VOICE TRANSCRIPT:\n{transcript}\n\n"
    if not combined_text:
        combined_text = "No text description provided."

    parts = [combined_text]

    if image_pil is not None:
        buf = io.BytesIO()
        image_pil.convert("RGB").save(buf, format="JPEG")  # JPEG can't hold alpha/palette modes (RGBA/P PNGs)
        parts.append(
            types.Part.from_bytes(
                data=buf.getvalue(),
                mime_type="image/jpeg",
            )
        )

    parts.append("Analyze the above and return the structured JSON as specified.")

    for attempt in range(2):
        try:
            response = _generate(
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=build_analysis_prompt(),
                    temperature=0.1,
                ),
            )
            data = _parse_json_response(response.text)
            return StructuredIssue(**data)
        except Exception as e:
            logger.warning(f"Analysis attempt {attempt + 1} failed: {e}")
            if attempt == 1:
                raise RuntimeError(f"Gemini analysis failed after 2 attempts: {e}")


def run_contradiction_audit(issue: StructuredIssue) -> ContradictionAudit:
    user_claims = issue.user_reported_facts + issue.inferred_information
    if not user_claims:
        return ContradictionAudit(
            flags=[],
            verified_facts=issue.observed_evidence,
            audit_passed=True,
        )

    image_evidence = "\n".join(issue.observed_evidence) if issue.observed_evidence else "No image provided."
    claims_text = "\n".join(f"- {c}" for c in user_claims)

    prompt = CONTRADICTION_PROMPT.format(
        user_claims=claims_text,
        image_evidence=image_evidence,
    )

    for attempt in range(2):
        try:
            response = _generate(
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            data = _parse_json_response(response.text)
            flags = [ContradictionFlag(**f) for f in data.get("flags", [])]

            verified = [f.claim for f in flags if f.status == "supported"] + issue.observed_evidence
            contradicted = [f.claim for f in flags if f.status == "contradicted"]
            unverifiable = [f.claim for f in flags if f.status == "unverifiable"]

            return ContradictionAudit(
                flags=flags,
                verified_facts=verified,
                contradicted_claims=contradicted,
                unverifiable_claims=unverifiable,
                contradiction_count=len(contradicted),
                audit_passed=len(contradicted) == 0,
            )
        except Exception as e:
            logger.warning(f"Contradiction audit attempt {attempt + 1} failed: {e}")
            if attempt == 1:
                return ContradictionAudit(
                    flags=[],
                    verified_facts=issue.observed_evidence + issue.user_reported_facts,
                    audit_passed=True,
                )


def generate_complaint(issue: StructuredIssue, audit: ContradictionAudit) -> str:
    prompt = COMPLAINT_GENERATION_PROMPT.format(
        structured_issue=issue.model_dump_json(indent=2),
        verified_facts="\n".join(f"- {f}" for f in audit.verified_facts),
    )

    response = _generate(
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return response.text.strip()
