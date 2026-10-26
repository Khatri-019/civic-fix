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
