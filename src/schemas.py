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
