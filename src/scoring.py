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
