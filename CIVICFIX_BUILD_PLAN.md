# CivicFix — Phase-Gated Build Plan for Claude Code

> **How to use this file:**
> Read and execute ONE phase at a time. After completing a phase, stop and wait for manual approval before reading the next phase. Do not read ahead. Each phase has explicit acceptance criteria — do not move on until all criteria pass.

---

## Project Overview

**CivicFix** is a multimodal Generative AI system that converts unstructured civic complaints (text + voice + image) into structured, validated, actionable reports.

**Core research objectives:**
1. Multimodal structuring pipeline using pre-trained AND custom-trained models
2. Contradiction detection — cross-check user claims against image evidence
3. Hallucination reduction — measure unsupported-sentence rate with/without audit stage
4. Measurable evaluation — modality ablation, classification accuracy, contradiction P/R, deterministic quality score

**Model strategy:**
- **Gemini multimodal API** — image understanding, structured extraction, complaint generation
- **Fine-tuned DistilBERT** — category + severity classification (custom model)
- **Cross-encoder MiniLM-L6** — contradiction detection via NLI (custom model)
- Both custom models trained on synthetic data; stubs ready for real data swap

**Stack:** Python 3.10+, Streamlit, Google Gemini API, Whisper, DistilBERT, MiniLM-L6, Pydantic, HuggingFace Transformers, scikit-learn, pandas

---

## Environment Variables

Create `.env` from `.env.example`:

```
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
```

Never hard-code keys. Never commit `.env`.

---

## Project Structure (target)

```
civicfix/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── CIVICFIX_BUILD_PLAN.md
│
├── src/
│   ├── __init__.py
│   ├── ai_service.py          # Gemini calls
│   ├── speech_service.py      # Whisper transcription
│   ├── schemas.py             # Pydantic models
│   ├── prompts.py             # All prompts isolated here
│   ├── scoring.py             # Deterministic quality score
│   ├── contradiction.py       # Contradiction audit stage
│   └── utils.py
│
├── models/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic dataset generation
│   ├── classifier.py          # DistilBERT fine-tuning + inference
│   └── contradiction_model.py # MiniLM-L6 NLI fine-tuning + inference
│
├── eval/
│   ├── __init__.py
│   ├── eval_runner.py         # Full evaluation suite
│   ├── metrics.py             # All metric calculations
│   └── results/               # JSON + CSV eval outputs
│
├── data/
│   ├── synthetic/             # Generated training data
│   └── real/                  # Placeholder for real data (drop CSV here)
│
└── sample_data/
    ├── broken_light.jpg
    ├── garbage.jpg
    └── water_leak.jpg
```

---

---

# PHASE 0 — Project Setup

**Read this phase. Complete it. Stop.**

## Tasks

1. Create the full directory structure above.
2. Create `requirements.txt`:

```
streamlit>=1.35.0
google-genai>=0.8.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pillow>=10.0.0
openai-whisper>=20231117
torch>=2.0.0
transformers>=4.40.0
datasets>=2.19.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
sentence-transformers>=3.0.0
evaluate>=0.4.0
seaborn>=0.13.0
matplotlib>=3.8.0
```

3. Create `.env.example`:

```
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
```

4. Create `.gitignore`:

```
.env
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/
dist/
build/
.DS_Store
eval/results/*.json
eval/results/*.csv
models/saved/
data/synthetic/*.csv
data/real/*.csv
```

5. Create `config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

CATEGORIES = [
    "Electrical",
    "Plumbing / Water",
    "Sanitation",
    "Waste Management",
    "Infrastructure",
    "Security",
    "Cleaning / Housekeeping",
    "Furniture / Facilities",
    "Internet / Network",
    "Transport",
    "Other",
]

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

DEPARTMENTS = {
    "Electrical": "Electrical Maintenance",
    "Plumbing / Water": "Plumbing / Facilities",
    "Sanitation": "Housekeeping / Sanitation",
    "Waste Management": "Waste Management",
    "Internet / Network": "IT Support",
    "Furniture / Facilities": "Facilities / Maintenance",
    "Security": "Security Office",
    "Cleaning / Housekeeping": "Housekeeping / Sanitation",
    "Infrastructure": "Civil / Maintenance",
    "Transport": "Transport Office",
    "Other": "General Maintenance",
}
```

6. Create a minimal `app.py` that only renders a header and confirms Streamlit works:

```python
import streamlit as st

st.set_page_config(page_title="CivicFix", page_icon="🏛️", layout="wide")
st.title("🏛️ CivicFix")
st.caption("Turn everyday problems into actionable reports.")
st.info("Phase 0 complete — environment ready.")
```

7. Create all `__init__.py` files (empty).

8. Run `pip install -r requirements.txt` in the virtual environment.

9. Run `streamlit run app.py` and verify the page loads.

## Acceptance Criteria

- [ ] `streamlit run app.py` opens without errors
- [ ] All directories exist
- [ ] `.env.example` present, no `.env` committed
- [ ] `config.py` imports cleanly: `python -c "import config; print(config.CATEGORIES)"`
- [ ] `requirements.txt` installs without conflicts

**STOP. Report results. Wait for approval.**

---

---

# PHASE 1 — Streamlit UI (No AI)

**Read this phase only after Phase 0 is approved.**

## Tasks

Build the full UI shell with all inputs wired up but no AI calls yet. Use `st.session_state` for all state.

### `app.py` — Full UI shell

Build these sections:

**Header:**
```
🏛️ CivicFix
Turn everyday problems into actionable reports.
AI-powered multimodal civic issue reporting
```

**Input section (left column, ~40% width):**
- `st.text_area` — "Describe the problem" (supports Hinglish, informal language)
- `st.file_uploader` — "Upload Voice Recording" (accepts `.wav`, `.mp3`, `.m4a`, `.ogg`)
- `st.file_uploader` — "Upload Image" (accepts `.jpg`, `.jpeg`, `.png`, `.webp`)
- If image uploaded: display it with `st.image`
- `st.button("🔍 Analyze Issue")` — disabled if all three inputs are empty

**Results section (right column, ~60% width):**
- Show placeholder cards when no analysis has run yet
- "Reset" button clears all session state

**Processing status placeholder:**
- Use `st.status` or `st.empty` — will be populated in Phase 3

**Footer:**
```
⚠️ Uploaded media is used for analysis only. Avoid submitting sensitive personal information.
Severity and category are AI-assisted estimates, not professional assessments.
```

### `src/utils.py`

```python
import base64
from pathlib import Path
from PIL import Image
import io


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def save_temp_audio(uploaded_file) -> Path:
    import tempfile
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return Path(f.name)


def load_image_pil(uploaded_file) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_file.read()))
```

## Acceptance Criteria

- [ ] All three inputs render and accept content
- [ ] Image displays after upload
- [ ] Analyze button is disabled when all inputs empty
- [ ] Reset clears all inputs and results
- [ ] No crashes on any input combination
- [ ] No AI calls made in this phase

**STOP. Report results. Wait for approval.**

---

---

# PHASE 2 — Speech-to-Text (Whisper)

**Read this phase only after Phase 1 is approved.**

## Tasks

### `src/speech_service.py`

```python
import whisper
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def transcribe_audio(audio_path: Path) -> dict:
    """
    Returns:
        {
            "success": bool,
            "transcript": str | None,
            "language": str | None,
            "error": str | None
        }
    """
    try:
        model = _get_model()
        result = model.transcribe(str(audio_path), fp16=False)
        return {
            "success": True,
            "transcript": result["text"].strip(),
            "language": result.get("language"),
            "error": None,
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {
            "success": False,
            "transcript": None,
            "language": None,
            "error": str(e),
        }
```

### Wire into `app.py`

When audio is uploaded and Analyze is clicked:
- Save to temp file via `save_temp_audio`
- Call `transcribe_audio`
- Display transcript in a collapsible `st.expander("📝 Voice Transcript")`
- If transcription fails: show `st.warning("Voice transcription failed — continuing with text/image.")` — do not crash
- Clean up temp file after transcription

## Acceptance Criteria

- [ ] `.wav` file transcribes correctly
- [ ] Transcript displays in expander
- [ ] App does not crash if transcription fails
- [ ] App does not crash if no audio uploaded
- [ ] Temp file is cleaned up after use
- [ ] Test: upload silent/corrupt audio — graceful error shown

**STOP. Report results. Wait for approval.**

---

---

# PHASE 3 — Pydantic Schemas

**Read this phase only after Phase 2 is approved.**

## Tasks

### `src/schemas.py`

Define all data models. These are the contracts between every module.

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


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

## Acceptance Criteria

- [ ] `from src.schemas import CivicFixReport` imports cleanly
- [ ] All enums and validators work: `python -c "from src.schemas import SeverityLevel; print(list(SeverityLevel))"`
- [ ] Invalid severity value raises `ValidationError`
- [ ] `confidence` outside [0,1] raises `ValidationError`

**STOP. Report results. Wait for approval.**

---

---

# PHASE 4 — Gemini Multimodal Analysis

**Read this phase only after Phase 3 is approved.**

## Tasks

### `src/prompts.py`

Store ALL prompts here. No prompt strings anywhere else.

```python
from config import CATEGORIES, SEVERITY_LEVELS

ANALYSIS_SYSTEM_PROMPT = """You are CivicFix, a multimodal civic issue analysis assistant.

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
"""

COMPLAINT_GENERATION_PROMPT = """You are generating a formal civic complaint report.

Use ONLY the verified facts provided below. Do not add any information not present in the structured issue.
Do not speculate. Do not use phrases like "it appears" or "possibly" for facts — state verified facts directly.
For genuinely uncertain items, omit them rather than hedging.

Write in formal English. 3-5 sentences. Professional tone.

STRUCTURED ISSUE:
{structured_issue}

VERIFIED FACTS (from contradiction audit):
{verified_facts}

Generate the complaint now. Return ONLY the complaint text, no JSON, no labels.
"""

CONTRADICTION_PROMPT = """You are a fact-checking assistant for civic complaints.

Compare each USER CLAIM against the IMAGE EVIDENCE and TEXT CONTEXT.
For each claim, determine:
- "supported": the image or context clearly supports this claim
- "contradicted": the image or context contradicts this claim
- "unverifiable": cannot be confirmed or denied from available evidence

USER CLAIMS:
{user_claims}

IMAGE EVIDENCE (what is actually visible):
{image_evidence}

Return ONLY valid JSON:
{{
  "flags": [
    {{
      "claim": "the user claim",
      "status": "supported | contradicted | unverifiable",
      "reason": "one sentence explanation"
    }}
  ]
}}
"""


def build_analysis_prompt() -> str:
    return ANALYSIS_SYSTEM_PROMPT.format(
        categories=", ".join(CATEGORIES),
        severity_levels=", ".join(SEVERITY_LEVELS),
    )
```

### `src/ai_service.py`

```python
import json
import logging
from typing import Optional
from google import genai
from google.genai import types
from PIL import Image
import io

from config import GEMINI_API_KEY, GEMINI_MODEL
from src.schemas import StructuredIssue, ContradictionAudit, ContradictionFlag
from src.prompts import build_analysis_prompt, COMPLAINT_GENERATION_PROMPT, CONTRADICTION_PROMPT

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


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
    client = _get_client()

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
        image_pil.save(buf, format="JPEG")
        parts.append(
            types.Part.from_bytes(
                data=buf.getvalue(),
                mime_type="image/jpeg",
            )
        )

    parts.append("Analyze the above and return the structured JSON as specified.")

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
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
    client = _get_client()

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
            response = client.models.generate_content(
                model=GEMINI_MODEL,
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
    client = _get_client()

    prompt = COMPLAINT_GENERATION_PROMPT.format(
        structured_issue=issue.model_dump_json(indent=2),
        verified_facts="\n".join(f"- {f}" for f in audit.verified_facts),
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return response.text.strip()
```

### Wire into `app.py`

When Analyze is clicked, show a `st.status` block with steps:
1. ✓ Input received
2. ✓ Speech transcribed (if audio)
3. ✓ Image analyzed
4. ✓ Issue classified
5. ✓ Contradiction audit complete
6. ✓ Complaint generated

Display the `StructuredIssue` fields in the results panel:
- Problem, Category, Severity (colored badge: Low=green, Medium=yellow, High=orange, Critical=red)
- Severity reason
- Department
- Location / Duration (show "Not specified" if null)
- Four evidence lists in expanders
- Contradiction audit summary (how many claims verified/contradicted/unverifiable)
- Generated complaint in `st.text_area` (editable)
- Copy button using `st.code` + clipboard trick

## Acceptance Criteria

- [ ] Text-only input produces valid `StructuredIssue`
- [ ] Image-only input produces valid `StructuredIssue`
- [ ] Text + image produces valid `StructuredIssue`
- [ ] Contradiction audit runs and returns flags
- [ ] Complaint generates and displays in editable text area
- [ ] API key error shows friendly message, not stack trace
- [ ] Pydantic validates output — malformed JSON retries once
- [ ] All severity badges show correct colors

**STOP. Report results. Wait for approval.**

---

---

# PHASE 5 — Deterministic Quality Score

**Read this phase only after Phase 4 is approved.**

## Tasks

### `src/scoring.py`

The quality score must be deterministic (same inputs → same score, no LLM involved).

```python
from src.schemas import StructuredIssue, ContradictionAudit, QualityScore
import re


def score_raw_input(text: str, has_image: bool, has_audio: bool) -> float:
    """Score the raw unstructured input before AI processing."""
    score = 0.0
    if text:
        word_count = len(text.split())
        score += min(word_count / 20, 1.0) * 30   # max 30 for text length
        if any(loc in text.lower() for loc in ["block", "hostel", "outside", "near", "floor", "room", "gate", "road"]):
            score += 15
        if any(dur in text.lower() for dur in ["day", "week", "hour", "since", "din", "ghante", "hafte"]):
            score += 10
    if has_image:
        score += 25
    if has_audio:
        score += 20
    return min(round(score, 1), 100.0)


def score_completeness(issue: StructuredIssue) -> float:
    score = 0.0
    if issue.problem:
        score += 25
    if issue.category:
        score += 20
    if issue.location:
        score += 25
    if issue.duration:
        score += 15
    if issue.department:
        score += 15
    return min(score, 100.0)


def score_specificity(issue: StructuredIssue) -> float:
    score = 0.0
    if issue.problem and len(issue.problem.split()) >= 4:
        score += 30
    if issue.subcategory:
        score += 20
    if issue.severity_reason and len(issue.severity_reason.split()) >= 5:
        score += 25
    if issue.recommended_action and len(issue.recommended_action.split()) >= 5:
        score += 25
    return min(score, 100.0)


def score_evidence(issue: StructuredIssue, audit: ContradictionAudit) -> float:
    score = 0.0
    if issue.observed_evidence:
        score += min(len(issue.observed_evidence) * 20, 50)
    if issue.user_reported_facts:
        score += min(len(issue.user_reported_facts) * 10, 30)
    if audit.contradiction_count == 0:
        score += 20
    else:
        score -= audit.contradiction_count * 10
    return max(min(score, 100.0), 0.0)


def score_actionability(issue: StructuredIssue) -> float:
    score = 0.0
    if issue.department:
        score += 30
    if issue.recommended_action:
        score += 35
    if issue.severity in ("High", "Critical"):
        score += 20
    if not issue.missing_information:
        score += 15
    return min(score, 100.0)


def calculate_quality_score(
    issue: StructuredIssue,
    audit: ContradictionAudit,
    raw_text: str,
    has_image: bool,
    has_audio: bool,
) -> QualityScore:
    completeness = score_completeness(issue)
    specificity = score_specificity(issue)
    evidence = score_evidence(issue, audit)
    actionability = score_actionability(issue)
    overall = round(
        completeness * 0.25 + specificity * 0.25 + evidence * 0.25 + actionability * 0.25, 1
    )
    raw_score = score_raw_input(raw_text or "", has_image, has_audio)

    return QualityScore(
        completeness=completeness,
        specificity=specificity,
        evidence=evidence,
        actionability=actionability,
        overall=overall,
        raw_input_score=raw_score,
    )


def count_unsupported_sentences(complaint: str, verified_facts: list[str]) -> tuple[int, int]:
    """
    Returns (unsupported_count, total_count).
    A sentence is considered unsupported if none of the verified facts
    share at least 2 content words with it.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]", complaint) if s.strip()]
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
                 "it", "its", "this", "that", "and", "or", "but", "in", "on", "at",
                 "to", "for", "of", "with", "by", "from", "be", "been", "being"}

    unsupported = 0
    for sentence in sentences:
        s_words = set(sentence.lower().split()) - stopwords
        supported = False
        for fact in verified_facts:
            f_words = set(fact.lower().split()) - stopwords
            if len(s_words & f_words) >= 2:
                supported = True
                break
        if not supported:
            unsupported += 1

    return unsupported, len(sentences)
```

### Wire into `app.py`

Display quality score after results:

```
Complaint Quality Score
Before (raw input):  35 / 100
After (AI report):   88 / 100

Completeness    88 / 100  ████████░░
Specificity     85 / 100  ████████░░
Evidence        92 / 100  █████████░
Actionability   87 / 100  ████████░░

Unsupported sentences: 1 / 5
```

Use `st.progress` for the bars. Show before vs after clearly.

## Acceptance Criteria

- [ ] Score is deterministic — same input always gives same score
- [ ] Raw input score is visibly lower than AI report score in all demo cases
- [ ] Unsupported sentence count is correct for a manually checked example
- [ ] Score displays with progress bars in UI

**STOP. Report results. Wait for approval.**

---

---

# PHASE 6 — Synthetic Data Generation

**Read this phase only after Phase 5 is approved.**

## Tasks

Generate a labeled dataset for training the custom models.

### `models/data_generator.py`

```python
"""
Generate synthetic CivicFix training data using Gemini.
Run: python -m models.data_generator
Output: data/synthetic/complaints.csv
"""
import json
import time
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, CATEGORIES, SEVERITY_LEVELS

client = genai.Client(api_key=GEMINI_API_KEY)

GENERATION_PROMPT = """Generate {n} realistic civic complaint examples for a college/hostel environment.

For each example, provide:
1. raw_complaint: informal text (1-2 sentences, may include Hinglish, typos, vague language)
2. category: one of {categories}
3. severity: one of {severity_levels}
4. has_contradiction: true if the complaint contains an exaggerated or unverifiable claim, false otherwise

Return ONLY a JSON array. No markdown.

Example entry:
{{
  "raw_complaint": "light kharab hai hostel ke bahar 3 din se",
  "category": "Electrical",
  "severity": "High",
  "has_contradiction": false
}}
"""


def generate_batch(n: int = 20) -> list[dict]:
    prompt = GENERATION_PROMPT.format(
        n=n,
        categories=", ".join(CATEGORIES),
        severity_levels=", ".join(SEVERITY_LEVELS),
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.9),
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_dataset(total: int = 200, batch_size: int = 20) -> pd.DataFrame:
    all_samples = []
    batches = total // batch_size
    print(f"Generating {total} samples in {batches} batches...")

    for i in range(batches):
        print(f"  Batch {i + 1}/{batches}...")
        try:
            batch = generate_batch(batch_size)
            all_samples.extend(batch)
            time.sleep(1.5)  # rate limit
        except Exception as e:
            print(f"  Batch {i + 1} failed: {e}")
            continue

    df = pd.DataFrame(all_samples)
    # Normalize category labels
    df = df[df["category"].isin(CATEGORIES)]
    df = df[df["severity"].isin(SEVERITY_LEVELS)]
    df = df.dropna(subset=["raw_complaint", "category", "severity"])
    df = df.drop_duplicates(subset=["raw_complaint"])
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    output_path = Path("data/synthetic/complaints.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(total=200, batch_size=20)
    df.to_csv(output_path, index=False)
    print(f"\nGenerated {len(df)} samples → {output_path}")
    print(df["category"].value_counts())
    print(df["severity"].value_counts())
```

Run it: `python -m models.data_generator`

## Acceptance Criteria

- [ ] Script runs without errors
- [ ] `data/synthetic/complaints.csv` exists with 150+ rows (some batches may fail)
- [ ] All categories appear in the dataset
- [ ] All severity levels appear in the dataset
- [ ] No empty `raw_complaint` values
- [ ] Print label distribution after generation

**STOP. Report results. Wait for approval.**

---

---

# PHASE 7 — Custom Model Training (DistilBERT + MiniLM)

**Read this phase only after Phase 6 is approved.**

## Tasks

Train two custom models on the synthetic dataset.

### `models/classifier.py`

Fine-tune DistilBERT for dual-label classification (category + severity).

```python
"""
Train and serve a DistilBERT classifier for complaint category + severity.
Training: python -m models.classifier --train
Inference: imported as a module
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import evaluate

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = Path("models/saved/classifier")
LABEL_DIR = Path("models/saved/labels")


class ComplaintDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_data(csv_path: Path, label_col: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["raw_complaint", label_col])
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col])
    return df["raw_complaint"].tolist(), labels, le


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(label_col: str, csv_path: Path):
    texts, labels, le = load_data(csv_path, label_col)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    val_enc = tokenizer(X_val, truncation=True, padding=True, max_length=128)

    train_dataset = ComplaintDataset(train_enc, y_train)
    val_dataset = ComplaintDataset(val_enc, y_val)

    save_path = SAVE_DIR / label_col
    save_path.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    )

    args = TrainingArguments(
        output_dir=str(save_path),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=str(save_path / "logs"),
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    # Save label mapping
    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(LABEL_DIR / f"{label_col}_labels.json", "w") as f:
        json.dump(label_map, f)

    # Eval report
    preds = np.argmax(trainer.predict(val_dataset).predictions, axis=-1)
    print(f"\n=== {label_col} Classification Report ===")
    print(classification_report(y_val, preds, target_names=le.classes_))

    return str(save_path)


_loaded_models = {}


def predict(text: str, label_col: str) -> str:
    global _loaded_models
    if label_col not in _loaded_models:
        save_path = SAVE_DIR / label_col
        if not save_path.exists():
            return None
        from transformers import pipeline
        _loaded_models[label_col] = {
            "pipe": pipeline("text-classification", model=str(save_path), device=-1),
            "labels": json.load(open(LABEL_DIR / f"{label_col}_labels.json")),
        }
    entry = _loaded_models[label_col]
    result = entry["pipe"](text, truncation=True, max_length=128)[0]
    label_id = int(result["label"].replace("LABEL_", ""))
    return entry["labels"][label_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", default="data/synthetic/complaints.csv")
    args = parser.parse_args()

    if args.train:
        csv = Path(args.data)
        print("Training category classifier...")
        train("category", csv)
        print("\nTraining severity classifier...")
        train("severity", csv)
        print("\nDone. Models saved to models/saved/classifier/")
```

Run: `python -m models.classifier --train`

### `models/contradiction_model.py`

Use MiniLM cross-encoder for NLI-based contradiction detection.

```python
"""
Contradiction detection using sentence-transformers cross-encoder (MiniLM NLI).
This uses the pre-trained cross-encoder/nli-MiniLM2-L6-H768 model.
Fine-tuning on synthetic data if contradiction pairs are available.

Usage: imported as module
"""
from sentence_transformers import CrossEncoder
from pathlib import Path
import pandas as pd

MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
SAVE_DIR = Path("models/saved/contradiction")

_model = None


def _get_model():
    global _model
    if _model is None:
        local = SAVE_DIR
        if local.exists():
            _model = CrossEncoder(str(local))
        else:
            _model = CrossEncoder(MODEL_NAME)
    return _model


def check_contradiction(claim: str, evidence: str) -> dict:
    """
    Returns:
        {
            "label": "entailment" | "neutral" | "contradiction",
            "score": float,
            "status": "supported" | "unverifiable" | "contradicted"
        }
    """
    model = _get_model()
    # Cross-encoder NLI: [premise, hypothesis]
    scores = model.predict([(evidence, claim)])
    # Labels for this model: contradiction=0, entailment=1, neutral=2
    label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
    label_idx = int(scores.argmax())
    label = label_map[label_idx]

    status_map = {
        "entailment": "supported",
        "neutral": "unverifiable",
        "contradiction": "contradicted",
    }

    return {
        "label": label,
        "score": float(scores.max()),
        "status": status_map[label],
    }


def batch_check(claims: list[str], evidence_text: str) -> list[dict]:
    model = _get_model()
    pairs = [(evidence_text, claim) for claim in claims]
    scores_batch = model.predict(pairs)
    label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
    status_map = {"entailment": "supported", "neutral": "unverifiable", "contradiction": "contradicted"}

    results = []
    for i, scores in enumerate(scores_batch):
        label_idx = int(scores.argmax())
        label = label_map[label_idx]
        results.append({
            "claim": claims[i],
            "label": label,
            "score": float(scores.max()),
            "status": status_map[label],
        })
    return results
```

### Update `src/contradiction.py`

```python
"""
Hybrid contradiction audit:
- Stage 1: Gemini-based (from Phase 4, ai_service.run_contradiction_audit)
- Stage 2: MiniLM cross-encoder NLI
Returns merged ContradictionAudit.
"""
from src.schemas import StructuredIssue, ContradictionAudit, ContradictionFlag
from models.contradiction_model import batch_check


def run_nli_audit(issue: StructuredIssue) -> ContradictionAudit:
    user_claims = issue.user_reported_facts + issue.inferred_information
    evidence_text = " ".join(issue.observed_evidence) if issue.observed_evidence else "No image evidence available."

    if not user_claims:
        return ContradictionAudit(
            flags=[],
            verified_facts=issue.observed_evidence,
            audit_passed=True,
        )

    results = batch_check(user_claims, evidence_text)

    flags = [
        ContradictionFlag(claim=r["claim"], status=r["status"], reason=f"NLI score: {r['score']:.2f}")
        for r in results
    ]
    verified = [r["claim"] for r in results if r["status"] == "supported"] + issue.observed_evidence
    contradicted = [r["claim"] for r in results if r["status"] == "contradicted"]
    unverifiable = [r["claim"] for r in results if r["status"] == "unverifiable"]

    return ContradictionAudit(
        flags=flags,
        verified_facts=verified,
        contradicted_claims=contradicted,
        unverifiable_claims=unverifiable,
        contradiction_count=len(contradicted),
        audit_passed=len(contradicted) == 0,
    )
```

### Wire custom models into `app.py`

After Gemini analysis, run:
1. DistilBERT category prediction → store as `custom_category`
2. DistilBERT severity prediction → store as `custom_severity`
3. MiniLM NLI audit → store as second audit result alongside Gemini audit

Display in results:

```
Classification
                Gemini      Custom (DistilBERT)
Category:       Electrical  Electrical
Severity:       High        High
```

Show both audits side by side (or togglable).

## Acceptance Criteria

- [ ] DistilBERT trains without OOM errors (if GPU unavailable, CPU is fine — will be slow)
- [ ] `models/saved/classifier/category/` and `/severity/` exist after training
- [ ] `predict("light kharab hai", "category")` returns a valid category
- [ ] MiniLM loads and `check_contradiction("light has been broken for 3 days", "A streetlight fixture is visible")` returns a dict
- [ ] Both model outputs appear in UI

**STOP. Report results. Wait for approval.**

---

---

# PHASE 8 — Evaluation Suite

**Read this phase only after Phase 7 is approved.**

## Tasks

Build a complete, reproducible evaluation that produces a JSON report.

### `eval/metrics.py`

```python
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import re


def classification_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "weighted_precision": round(float(p), 4),
        "weighted_recall": round(float(r), 4),
        "weighted_f1": round(float(f1), 4),
        "per_class": report,
    }


def contradiction_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """
    y_true / y_pred: list of "supported" | "contradicted" | "unverifiable"
    Returns precision and recall for the "contradicted" class.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "contradicted" and p == "contradicted")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "contradicted" and p == "contradicted")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "contradicted" and p != "contradicted")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "contradiction_precision": round(precision, 4),
        "contradiction_recall": round(recall, 4),
        "contradiction_f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def unsupported_sentence_rate(complaint: str, verified_facts: list[str]) -> dict:
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
                 "it", "its", "this", "that", "and", "or", "but", "in", "on", "at",
                 "to", "for", "of", "with", "by", "from", "be", "been", "being"}
    sentences = [s.strip() for s in re.split(r"[.!?]", complaint) if s.strip()]
    unsupported = 0
    for sentence in sentences:
        s_words = set(sentence.lower().split()) - stopwords
        supported = any(
            len(s_words & (set(f.lower().split()) - stopwords)) >= 2
            for f in verified_facts
        )
        if not supported:
            unsupported += 1
    total = len(sentences)
    rate = unsupported / total if total > 0 else 0.0
    return {
        "unsupported_sentences": unsupported,
        "total_sentences": total,
        "unsupported_rate": round(rate, 4),
    }
```

### `eval/eval_runner.py`

```python
"""
Full evaluation suite for CivicFix.
Run: python -m eval.eval_runner
Output: eval/results/eval_report.json
"""
import json
import time
from pathlib import Path
from itertools import combinations
import pandas as pd

from src.ai_service import analyze_issue, run_contradiction_audit, generate_complaint
from src.scoring import calculate_quality_score, count_unsupported_sentences
from models.classifier import predict as custom_predict
from models.contradiction_model import batch_check
from eval.metrics import classification_metrics, contradiction_metrics, unsupported_sentence_rate
from config import CATEGORIES, SEVERITY_LEVELS

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hand-label these 15 test cases — edit after generation
TEST_CASES = [
    {"text": "light kharab hai hostel ke bahar 3 din se", "category": "Electrical", "severity": "High", "has_contradiction": False},
    {"text": "garbage overflowing near block C since yesterday, smell is unbearable", "category": "Waste Management", "severity": "High", "has_contradiction": False},
    {"text": "water leaking from pipe in boys washroom second floor", "category": "Plumbing / Water", "severity": "Medium", "has_contradiction": False},
    {"text": "chair broken in classroom 204", "category": "Furniture / Facilities", "severity": "Low", "has_contradiction": False},
    {"text": "wifi not working in hostel for 2 days", "category": "Internet / Network", "severity": "Medium", "has_contradiction": False},
    {"text": "campus bus didnt come today morning", "category": "Transport", "severity": "Medium", "has_contradiction": False},
    {"text": "toilet not clean in ground floor washroom", "category": "Sanitation", "severity": "Medium", "has_contradiction": False},
    {"text": "electricity gone in entire block A since morning", "category": "Electrical", "severity": "Critical", "has_contradiction": False},
    {"text": "road outside college completely destroyed with 10 foot craters", "category": "Infrastructure", "severity": "High", "has_contradiction": True},  # exaggeration
    {"text": "drinking water contaminated definitely, everyone is getting sick", "category": "Plumbing / Water", "severity": "Critical", "has_contradiction": True},  # unverifiable claim
    {"text": "hostel gate lock broken anyone can enter", "category": "Security", "severity": "High", "has_contradiction": False},
    {"text": "sweeping not done in corridor for a week bad smell", "category": "Cleaning / Housekeeping", "severity": "Medium", "has_contradiction": False},
    {"text": "projector in seminar hall not working", "category": "Furniture / Facilities", "severity": "Medium", "has_contradiction": False},
    {"text": "street light outside gate is broken since 5 days very dangerous", "category": "Electrical", "severity": "High", "has_contradiction": False},
    {"text": "construction debris blocking main road completely impassable for ambulance", "category": "Infrastructure", "severity": "Critical", "has_contradiction": True},
]


MODALITY_SETS = [
    {"name": "text_only", "use_image": False, "use_audio": False},
    {"name": "image_only", "use_image": True, "use_audio": False},
    {"name": "text_image", "use_image": True, "use_audio": False},
    {"name": "text_audio", "use_image": False, "use_audio": True},
    {"name": "all_modalities", "use_image": True, "use_audio": True},
]


def run_eval():
    results = {
        "classification": {"gemini": [], "custom": [], "true": []},
        "severity": {"gemini": [], "custom": [], "true": []},
        "contradiction": {"gemini_pred": [], "custom_pred": [], "true": []},
        "quality_scores": [],
        "unsupported_rates": {"with_audit": [], "without_audit": []},
        "modality_ablation": {},
    }

    print(f"Running evaluation on {len(TEST_CASES)} test cases...")

    for i, case in enumerate(TEST_CASES):
        print(f"  Case {i + 1}/{len(TEST_CASES)}: {case['text'][:50]}...")
        try:
            issue = analyze_issue(text=case["text"], transcript=None, image_pil=None)
            audit = run_contradiction_audit(issue)
            complaint = generate_complaint(issue, audit)

            # Classification
            results["classification"]["gemini"].append(issue.category)
            results["classification"]["true"].append(case["category"])
            results["severity"]["gemini"].append(issue.severity.value)
            results["severity"]["true"].append(case["severity"])

            custom_cat = custom_predict(case["text"], "category")
            custom_sev = custom_predict(case["text"], "severity")
            results["classification"]["custom"].append(custom_cat or "Other")
            results["severity"]["custom"].append(custom_sev or "Medium")

            # Contradiction ground truth
            gt = "contradicted" if case["has_contradiction"] else "supported"
            results["contradiction"]["true"].append(gt)

            # Gemini audit contradiction label
            gemini_label = "contradicted" if audit.contradiction_count > 0 else "supported"
            results["contradiction"]["gemini_pred"].append(gemini_label)

            # NLI audit
            all_claims = issue.user_reported_facts + issue.inferred_information
            evidence = " ".join(issue.observed_evidence) if issue.observed_evidence else case["text"]
            if all_claims:
                nli_results = batch_check(all_claims, evidence)
                has_contradiction = any(r["status"] == "contradicted" for r in nli_results)
                results["contradiction"]["custom_pred"].append(
                    "contradicted" if has_contradiction else "supported"
                )
            else:
                results["contradiction"]["custom_pred"].append("supported")

            # Quality
            qs = calculate_quality_score(issue, audit, case["text"], False, False)
            results["quality_scores"].append(qs.model_dump())

            # Unsupported rate with audit
            ur_with = unsupported_sentence_rate(complaint, audit.verified_facts)
            results["unsupported_rates"]["with_audit"].append(ur_with["unsupported_rate"])

            # Without audit (use all facts including contradicted)
            all_facts = issue.user_reported_facts + issue.observed_evidence + issue.inferred_information
            ur_without = unsupported_sentence_rate(complaint, all_facts)
            results["unsupported_rates"]["without_audit"].append(ur_without["unsupported_rate"])

            time.sleep(0.5)

        except Exception as e:
            print(f"  Case {i + 1} failed: {e}")
            continue

    # Aggregate metrics
    cat_labels = CATEGORIES
    sev_labels = SEVERITY_LEVELS

    report = {
        "total_cases": len(TEST_CASES),
        "classification_accuracy": {
            "gemini": classification_metrics(
                results["classification"]["true"],
                results["classification"]["gemini"],
                cat_labels,
            ),
            "custom_distilbert": classification_metrics(
                results["classification"]["true"],
                results["classification"]["custom"],
                cat_labels,
            ),
        },
        "severity_accuracy": {
            "gemini": classification_metrics(
                results["severity"]["true"],
                results["severity"]["gemini"],
                sev_labels,
            ),
            "custom_distilbert": classification_metrics(
                results["severity"]["true"],
                results["severity"]["custom"],
                sev_labels,
            ),
        },
        "contradiction_detection": {
            "gemini": contradiction_metrics(
                results["contradiction"]["true"],
                results["contradiction"]["gemini_pred"],
            ),
            "custom_minilm": contradiction_metrics(
                results["contradiction"]["true"],
                results["contradiction"]["custom_pred"],
            ),
        },
        "hallucination_reduction": {
            "avg_unsupported_rate_with_audit": round(
                sum(results["unsupported_rates"]["with_audit"]) / max(len(results["unsupported_rates"]["with_audit"]), 1), 4
            ),
            "avg_unsupported_rate_without_audit": round(
                sum(results["unsupported_rates"]["without_audit"]) / max(len(results["unsupported_rates"]["without_audit"]), 1), 4
            ),
        },
        "avg_quality_score": round(
            sum(q["overall"] for q in results["quality_scores"]) / max(len(results["quality_scores"]), 1), 2
        ),
    }

    output_path = RESULTS_DIR / "eval_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete → {output_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_eval()
```

Run: `python -m eval.eval_runner`

## Acceptance Criteria

- [ ] `eval_runner.py` runs end-to-end without crashing
- [ ] `eval/results/eval_report.json` is generated
- [ ] All 4 metric categories appear in the report: classification, severity, contradiction, hallucination_reduction
- [ ] Gemini vs custom model comparison is visible in output
- [ ] Unsupported rate with audit is lower than without audit (or equal — never higher)

**STOP. Report results. Wait for approval.**

---

---

# PHASE 9 — UI Polish + Demo Preparation

**Read this phase only after Phase 8 is approved.**

## Tasks

### Final UI additions

1. **Before vs After comparison panel:**

```
Original Input                    AI-Structured Report
────────────────────────────────  ──────────────────────────────────
"light kharab hai hostel ke       Non-functional streetlight outside
 bahar"                           the hostel (Electrical / High)

Raw Quality: 35/100               AI Quality: 88/100
```

2. **Evaluation summary panel** (read from `eval/results/eval_report.json` if it exists):

```
Model Comparison
                     Gemini    DistilBERT
Category Accuracy    XX%       XX%
Severity Accuracy    XX%       XX%
Contradiction P/R    X.X/X.X   X.X/X.X

Hallucination Reduction
With audit:    X.X% unsupported
Without audit: X.X% unsupported
```

3. **Download Report button** — generates a plain text report:

```
CIVICFIX REPORT
===============
Problem: ...
Category: ...
Severity: ...
Department: ...

COMPLAINT:
...

QUALITY SCORE: 88/100
GENERATED: 2026-10-26
```

4. **Architecture diagram** in sidebar using `st.image` or `st.code`:

```
Voice → Whisper → Transcript
Image → Gemini → Evidence
Text + Transcript + Evidence → Gemini → Structured Issue
Structured Issue → DistilBERT → Category + Severity
User Claims + Evidence → MiniLM NLI → Contradiction Audit
Verified Facts → Gemini → Complaint
Complaint → Scoring → Quality Score
```

### Sample data

Add 3 images to `sample_data/`:
- `broken_light.jpg` — a streetlight or dark area
- `garbage.jpg` — overflowing bin
- `water_leak.jpg` — pipe or wet floor

Use royalty-free placeholder images if no real photos available.

Create `sample_data/voice_samples.md` with the 3 demo transcripts pre-written (for cases where live audio fails during demo).

### README.md

```markdown
# CivicFix

Multimodal GenAI civic complaint structuring system.

## Setup
pip install -r requirements.txt
cp .env.example .env  # add GEMINI_API_KEY

## Train custom models (optional — pre-trained fallback available)
python -m models.data_generator
python -m models.classifier --train

## Run
streamlit run app.py

## Evaluate
python -m eval.eval_runner
```

## Acceptance Criteria

- [ ] Three demo scenarios work end-to-end (broken light / garbage / water leak)
- [ ] Download report button produces a readable text file
- [ ] Before/after comparison visible
- [ ] Evaluation summary panel shows (even if eval_report.json doesn't exist — show "Run eval suite first")
- [ ] App handles missing model files gracefully (falls back to Gemini-only)

**STOP. Report results. Wait for approval.**

---

---

# PHASE 10 — Final Testing Checklist

**Read this phase only after Phase 9 is approved.**

## Manual Test Matrix

Run each case and record pass/fail:

| # | Input | Expected | Pass? |
|---|-------|----------|-------|
| 1 | Text only (English) | Valid report | |
| 2 | Text only (Hinglish) | Valid report | |
| 3 | Image only | Report with observed_evidence | |
| 4 | Audio only | Transcript shown + valid report | |
| 5 | Text + Image | Evidence from both sources | |
| 6 | Text + Audio | Transcript merged with text | |
| 7 | All three | Full multimodal report | |
| 8 | Vague input ("something broken") | Missing information listed | |
| 9 | Exaggerated claim + image | Contradiction flagged | |
| 10 | Empty submit | Friendly error, no crash | |
| 11 | Invalid image file | Graceful error |  |
| 12 | Corrupt audio | Graceful error, continue |  |
| 13 | API key missing | Friendly error message |  |
| 14 | Download report | Valid .txt file |  |
| 15 | Reset button | All state cleared |  |

## Acceptance Criteria

- [ ] All 15 test cases pass or have documented known failures
- [ ] No stack traces visible to user in any failure case
- [ ] Three primary demo scenarios work reliably back-to-back
- [ ] `python -m eval.eval_runner` produces report in under 5 minutes

---

---

# Real Data Swap Guide

When real complaint data is available, drop a CSV at `data/real/complaints.csv` with columns:

```
raw_complaint, category, severity, has_contradiction
```

Then retrain:

```bash
python -m models.classifier --train --data data/real/complaints.csv
```

No other code changes needed.

---

# Key Design Decisions

| Decision | Rationale |
|---|---|
| Gemini for extraction | Best multimodal capability, handles Hinglish |
| DistilBERT for classification | Fast, trainable on CPU, clear comparison point vs Gemini |
| MiniLM cross-encoder for NLI | Purpose-built for entailment/contradiction, no training needed |
| Deterministic quality score | Reproducible, explainable, not LLM-dependent |
| Two-stage complaint generation | Separates extraction from generation, reduces hallucination |
| Pydantic validation | Prevents malformed AI output from reaching UI |
| Synthetic dataset with swap stub | Demo works day 1, real data improves it later |
