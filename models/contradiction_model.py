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
