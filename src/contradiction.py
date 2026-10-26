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
