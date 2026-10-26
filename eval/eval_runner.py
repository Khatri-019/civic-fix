"""
Full evaluation suite for CivicFix.
Run: python -m eval.eval_runner
Output: eval/results/eval_report.json (+ eval/results/eval_cases.csv per-case details)

Gemini outputs are cached per case in eval/results/cache/ (free tier = 20 requests/day per model),
so an interrupted run resumes without re-spending quota. Delete the cache folder to re-query Gemini.
"""
import hashlib
import json
import time
from pathlib import Path
from itertools import combinations
import pandas as pd

import src.ai_service as ai_service
from src.ai_service import analyze_issue, run_contradiction_audit, generate_complaint
from src.schemas import StructuredIssue, ContradictionAudit
from src.scoring import calculate_quality_score, count_unsupported_sentences
from models.classifier import predict as custom_predict
from models.contradiction_model import batch_check
from eval.metrics import classification_metrics, contradiction_metrics, unsupported_sentence_rate
from config import CATEGORIES, SEVERITY_LEVELS

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hand-label these 15 test cases — edit after generation
TEST_CASES = [
    {"text": "light not working outside hostel since 3 days", "category": "Electrical", "severity": "High", "has_contradiction": False},
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


def normalize_category(gemini_category: str) -> str:
    """Gemini often returns "Main / Sub" (e.g. "Electrical / Lighting"); map it to the config category it starts with."""
    low = gemini_category.strip().lower()
    for cat in sorted(CATEGORIES, key=len, reverse=True):  # longest first: "Plumbing / Water" before shorter matches
        if low == cat.lower() or low.startswith(cat.lower() + " /") or low.startswith(cat.lower()):
            return cat
    return gemini_category


def _gemini_outputs(case: dict) -> tuple[StructuredIssue, ContradictionAudit, str, str, str]:
    """
    Run (or load cached) Gemini analysis → audit → complaint for one case, plus a second complaint generated
    WITHOUT the audit (every extracted fact passed through as "verified") for the hallucination ablation.
    """
    key = hashlib.sha1(case["text"].encode("utf-8")).hexdigest()[:12]
    path = CACHE_DIR / f"{key}.json"
    d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"text": case["text"]}

    if "issue" in d:
        issue, audit = StructuredIssue.model_validate(d["issue"]), ContradictionAudit.model_validate(d["audit"])
        complaint = d["complaint"]
    else:
        issue = analyze_issue(text=case["text"], transcript=None, image_pil=None)
        audit = run_contradiction_audit(issue)
        complaint = generate_complaint(issue, audit)
        d.update(issue=issue.model_dump(mode="json"), audit=audit.model_dump(), complaint=complaint,
                 model_used=ai_service.last_model_used)
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    if "complaint_no_audit" not in d:
        no_audit = ContradictionAudit(
            verified_facts=issue.user_reported_facts + issue.observed_evidence + issue.inferred_information,
            audit_passed=True,
        )
        d["complaint_no_audit"] = generate_complaint(issue, no_audit)
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    return issue, audit, complaint, d["complaint_no_audit"], d.get("model_used") or "cached"


def run_eval():
    results = {
        "classification": {"gemini": [], "custom": [], "true": []},
        "severity": {"gemini": [], "custom": [], "true": []},
        "contradiction": {"gemini_pred": [], "custom_pred": [], "true": []},
        "quality_scores": [],
        "unsupported_rates": {"with_audit": [], "without_audit": []},
        "modality_ablation": {},
    }
    case_rows = []
    failed_cases = []

    print(f"Running evaluation on {len(TEST_CASES)} test cases...")

    for i, case in enumerate(TEST_CASES):
        print(f"  Case {i + 1}/{len(TEST_CASES)}: {case['text'][:50]}...")
        try:
            issue, audit, complaint, complaint_no_audit, model_used = _gemini_outputs(case)

            # Classification
            gemini_cat = normalize_category(issue.category)
            results["classification"]["gemini"].append(gemini_cat)
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

            # Hallucination ablation: two complaints (generated with vs without the audit stage), both scored
            # against the SAME reference = what the input actually supports (raw text + facts extracted from it).
            # Gemini's own inferred_information is deliberately not in the reference.
            grounded_facts = [case["text"]] + issue.user_reported_facts + issue.observed_evidence
            ur_with = unsupported_sentence_rate(complaint, grounded_facts)
            results["unsupported_rates"]["with_audit"].append(ur_with["unsupported_rate"])
            ur_without = unsupported_sentence_rate(complaint_no_audit, grounded_facts)
            results["unsupported_rates"]["without_audit"].append(ur_without["unsupported_rate"])

            # Plan's original metric (same complaint vs two fact lists) — kept for transparency; by construction
            # it can never favour the audit, since audit.verified_facts is a subset of all extracted facts.
            all_facts = issue.user_reported_facts + issue.observed_evidence + issue.inferred_information
            results.setdefault("plan_metric", {"vs_verified": [], "vs_all_facts": []})
            results["plan_metric"]["vs_verified"].append(unsupported_sentence_rate(complaint, audit.verified_facts)["unsupported_rate"])
            results["plan_metric"]["vs_all_facts"].append(unsupported_sentence_rate(complaint, all_facts)["unsupported_rate"])

            case_rows.append({
                "case": i + 1, "text": case["text"], "model_used": model_used,
                "true_category": case["category"], "gemini_category_raw": issue.category,
                "gemini_category": gemini_cat, "custom_category": custom_cat,
                "true_severity": case["severity"], "gemini_severity": issue.severity.value, "custom_severity": custom_sev,
                "true_contradiction": gt, "gemini_contradiction": gemini_label,
                "custom_contradiction": results["contradiction"]["custom_pred"][-1],
                "quality_raw": qs.raw_input_score, "quality_overall": qs.overall,
                "unsupported_rate_with_audit": ur_with["unsupported_rate"],
                "unsupported_rate_without_audit": ur_without["unsupported_rate"],
                "complaint_with_audit": complaint,
                "complaint_without_audit": complaint_no_audit,
            })

            time.sleep(0.5)

        except Exception as e:
            print(f"  Case {i + 1} failed: {e}")
            failed_cases.append(i + 1)
            continue

    # Aggregate metrics
    cat_labels = CATEGORIES
    sev_labels = SEVERITY_LEVELS

    report = {
        "total_cases": len(TEST_CASES),
        "evaluated_cases": len(case_rows),
        "failed_cases": failed_cases,
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
            "method": "two complaints (with / without audit stage) scored against the same grounded reference: "
                      "raw input text + user_reported_facts + observed_evidence",
            "plan_original_metric_same_complaint": {
                "avg_rate_vs_audit_verified_facts": round(
                    sum(results.get("plan_metric", {}).get("vs_verified", [])) / max(len(results.get("plan_metric", {}).get("vs_verified", [])), 1), 4
                ),
                "avg_rate_vs_all_extracted_facts": round(
                    sum(results.get("plan_metric", {}).get("vs_all_facts", [])) / max(len(results.get("plan_metric", {}).get("vs_all_facts", [])), 1), 4
                ),
                "note": "same complaint vs two fact lists; verified facts are a subset of all facts, so this cannot favour the audit",
            },
        },
        "avg_quality_score": round(
            sum(q["overall"] for q in results["quality_scores"]) / max(len(results["quality_scores"]), 1), 2
        ),
    }

    output_path = RESULTS_DIR / "eval_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(case_rows).to_csv(RESULTS_DIR / "eval_cases.csv", index=False)

    print(f"\nEvaluation complete → {output_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_eval()
