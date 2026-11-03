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

TEXT_PRIORITY_COMPLAINT_PROMPT = """You are generating a formal civic complaint report.

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
"""


def build_analysis_prompt() -> str:
    return ANALYSIS_SYSTEM_PROMPT.format(
        categories=", ".join(CATEGORIES),
        severity_levels=", ".join(SEVERITY_LEVELS),
    )
