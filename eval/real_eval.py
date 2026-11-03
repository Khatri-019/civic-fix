"""
Evaluation on the real dataset (data/real/test.csv). The 15-case suite in eval_runner.py is kept unchanged.

Parts (run in this order; each Gemini answer is cached per case, so a quota stop resumes where it left off):
  classification - the 75 non-contradicting test rows, text only: Gemini (analyze_issue) vs DistilBERT,
                   category + severity. Contradicting rows are excluded: their severity is the photo's, not the text's.
  contradiction  - each test photo twice: with its contradicting text (positive) and with its informal consistent
                   text (control). Gemini uses the app's review rule (contradicted claims or image mismatch);
                   MiniLM NLI runs on the same extracted claims vs the photo evidence.

Run:    python -m eval.real_eval [--part classification|contradiction]
Output: eval/results/real_eval_report.json, eval/results/real_eval_cases.csv
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

import src.ai_service as ai_service
from src.ai_service import analyze_issue, run_contradiction_audit
from src.contradiction import run_nli_audit
from src.schemas import StructuredIssue, ContradictionAudit
from models.classifier import predict as custom_predict
from eval.eval_runner import normalize_category
from eval.metrics import classification_metrics, contradiction_metrics
from config import CATEGORIES, SEVERITY_LEVELS

TEST_CSV = Path("data/real/test.csv")
RESULTS_DIR = Path("eval/results")
CACHE_DIR = RESULTS_DIR / "real_cache"
REPORT = RESULTS_DIR / "real_eval_report.json"
CASES = RESULTS_DIR / "real_eval_cases.csv"
MAX_CONSECUTIVE_FAILURES = 3  # every model is probably out of quota; stop and resume later


class GeminiUnavailable(Exception):
    pass


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV, keep_default_na=False)
    df["has_contradiction"] = df.has_contradiction.astype(str).str.lower() == "true"
    return df


def cached(key: str, compute) -> dict:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = compute()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def gemini_text_only(row) -> dict:
    issue = analyze_issue(text=row.raw_complaint, transcript=None, image_pil=None)
    return {"issue": issue.model_dump(mode="json"), "model": ai_service.last_model_used}


def gemini_with_photo(row) -> dict:
    issue = analyze_issue(text=row.raw_complaint, transcript=None, image_pil=Image.open(row.image_file))
    model = ai_service.last_model_used
    ai_service.last_model_used = None
    audit = run_contradiction_audit(issue)
    # The audit swallows API errors and returns a "passed" fallback; caching that would turn a quota error
    # into a false negative. It made a real call only if last_model_used was set again.
    has_claims = bool(issue.user_reported_facts + issue.inferred_information)
    if has_claims and ai_service.last_model_used is None:
        raise GeminiUnavailable("contradiction audit got no Gemini answer")
    return {"issue": issue.model_dump(mode="json"), "audit": audit.model_dump(), "model": model}


def run_cases(rows: list, key_prefix: str, compute) -> dict:
    """Gemini answers per record_id; stops early after repeated failures (quota)."""
    answers, failures = {}, 0
    for number, row in enumerate(rows, start=1):
        try:
            answers[row.record_id] = cached(f"{key_prefix}_{row.record_id}", lambda: compute(row))
            failures = 0
        except Exception as e:  # analyze_issue wraps API errors in RuntimeError
            failures += 1
            print(f"  {row.record_id} failed: {str(e)[:140]}")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  Stopping after {failures} failures in a row (likely quota). Re-run later to resume.")
                break
        if number % 10 == 0 or number == len(rows):
            print(f"  {key_prefix}: {number}/{len(rows)} (answered {len(answers)})", flush=True)
    return answers


def wilson_interval(correct: int, total: int) -> list[float]:
    """95% confidence interval for an accuracy; with n = 75 the uncertainty is wide, so report it."""
    if total == 0:
        return [0.0, 0.0]
    z, p = 1.96, correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(centre - margin, 4), round(centre + margin, 4)]


def summarize(y_true: list, y_pred: list, labels: list) -> dict:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    metrics = classification_metrics(y_true, y_pred, labels)
    metrics.update(accuracy=round(accuracy_score(y_true, y_pred), 4), accuracy_ci95=wilson_interval(correct, len(y_true)),
                   macro_f1=round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
                   n=len(y_true))
    return metrics


def classification_part(test: pd.DataFrame, case_rows: list) -> dict:
    rows = list(test[~test.has_contradiction].itertuples())
    print(f"Classification: {len(rows)} text-only cases")
    answers = run_cases(rows, "text", gemini_text_only)
    truth = {"category": [], "severity": []}
    preds = {(s, c): [] for s in ("gemini", "custom") for c in ("category", "severity")}
    for row in rows:
        if row.record_id not in answers:
            continue
        issue = StructuredIssue.model_validate(answers[row.record_id]["issue"])
        gemini = {"category": normalize_category(issue.category), "severity": issue.severity.value}
        custom = {c: custom_predict(row.raw_complaint, c) for c in ("category", "severity")}
        for column in ("category", "severity"):
            truth[column].append(getattr(row, column))
            preds[("gemini", column)].append(gemini[column])
            preds[("custom", column)].append(custom[column])
        case_rows.append({"part": "classification", "record_id": row.record_id, "variant": row.variant,
                          "text": row.raw_complaint, "true_category": row.category, "true_severity": row.severity,
                          "gemini_category_raw": issue.category, "gemini_category": gemini["category"],
                          "gemini_severity": gemini["severity"], "custom_category": custom["category"],
                          "custom_severity": custom["severity"], "gemini_model": answers[row.record_id]["model"]})
    labels = {"category": CATEGORIES, "severity": SEVERITY_LEVELS}
    return {
        "cases_total": len(rows), "cases_evaluated": len(truth["category"]),
        **{f"{column}_accuracy": {
            "gemini": summarize(truth[column], preds[("gemini", column)], labels[column]),
            "custom_distilbert": summarize(truth[column], preds[("custom", column)], labels[column]),
        } for column in ("category", "severity")},
    }


def contradiction_part(test: pd.DataFrame, case_rows: list) -> dict:
    positives = test[test.has_contradiction & (test.image_file != "")]
    controls = test[(test.variant == "consistent_informal") & test.image_id.isin(positives.image_id)]
    rows = list(positives.itertuples()) + list(controls.itertuples())
    print(f"Contradiction: {len(positives)} contradicting + {len(controls)} control cases, each with its photo")
    answers = run_cases(rows, "photo", gemini_with_photo)
    truth, gemini_pred, minilm_pred = [], [], []
    for row in rows:
        if row.record_id not in answers:
            continue
        issue = StructuredIssue.model_validate(answers[row.record_id]["issue"])
        audit = ContradictionAudit.model_validate(answers[row.record_id]["audit"])
        nli = run_nli_audit(issue)
        label = lambda flagged: "contradicted" if flagged else "supported"
        truth.append(label(row.has_contradiction))
        gemini_pred.append(label(audit.contradiction_count > 0 or audit.image_mismatch))  # = app's review trigger
        minilm_pred.append(label(nli.contradiction_count > 0))
        case_rows.append({"part": "contradiction", "record_id": row.record_id, "variant": row.variant,
                          "text": row.raw_complaint, "image_file": row.image_file, "true_contradiction": truth[-1],
                          "gemini_contradiction": gemini_pred[-1], "minilm_contradiction": minilm_pred[-1],
                          "gemini_contradicted_claims": " | ".join(audit.contradicted_claims),
                          "gemini_image_mismatch": audit.image_mismatch, "gemini_mismatch_reason": audit.mismatch_reason,
                          "minilm_contradicted_claims": " | ".join(nli.contradicted_claims),
                          "gemini_model": answers[row.record_id]["model"]})
    def with_accuracy(pred):
        metrics = contradiction_metrics(truth, pred)
        metrics["accuracy"] = round(accuracy_score(truth, pred), 4) if truth else 0.0
        metrics["false_alarm_rate_on_controls"] = round(
            sum(p == "contradicted" and t == "supported" for t, p in zip(truth, pred))
            / max(truth.count("supported"), 1), 4)
        return metrics
    return {"cases_total": len(rows), "cases_evaluated": len(truth), "positives": truth.count("contradicted"),
            "controls": truth.count("supported"),
            "gemini": with_accuracy(gemini_pred), "custom_minilm": with_accuracy(minilm_pred)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["classification", "contradiction"], help="run only one part")
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    test = load_test()

    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    case_rows = []
    if args.part != "contradiction":
        report["classification"] = classification_part(test, case_rows)
    if args.part != "classification":
        report["contradiction_detection"] = contradiction_part(test, case_rows)
    report["dataset"] = f"{TEST_CSV.as_posix()} ({len(test)} rows)"

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    old = pd.read_csv(CASES, keep_default_na=False) if CASES.exists() else pd.DataFrame()
    parts_run = {r["part"] for r in case_rows}
    if len(old):
        old = old[~old.part.isin(parts_run)]
    pd.concat([old, pd.DataFrame(case_rows)]).to_csv(CASES, index=False)

    print(f"\n-> {REPORT.as_posix()}, {CASES.as_posix()}")
    if "classification" in report:
        c = report["classification"]
        print(f"Classification ({c['cases_evaluated']}/{c['cases_total']} cases)")
        for column in ("category", "severity"):
            for system, m in c[f"{column}_accuracy"].items():
                print(f"  {column:<9} {system:<18} acc {m['accuracy']:.3f} {m['accuracy_ci95']}  macro-F1 {m['macro_f1']:.3f}")
    if "contradiction_detection" in report:
        c = report["contradiction_detection"]
        print(f"Contradiction ({c['cases_evaluated']}/{c['cases_total']} cases: {c['positives']} positives, {c['controls']} controls)")
        for system in ("gemini", "custom_minilm"):
            m = c[system]
            print(f"  {system:<14} P {m['contradiction_precision']:.2f} R {m['contradiction_recall']:.2f} "
                  f"F1 {m['contradiction_f1']:.2f}  false alarms on controls {m['false_alarm_rate_on_controls']:.0%}")


if __name__ == "__main__":
    main()
