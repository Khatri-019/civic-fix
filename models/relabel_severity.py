"""
Re-label severity in the synthetic dataset with one consistent rubric.

Why: the generator assigned severity inconsistently across batches (e.g. "chair missing a leg" = Low,
"chair leg loose" = Medium), capping any text classifier at ~50% accuracy. This re-labels every row
with the same severity definitions the app's analysis prompt uses (src/prompts.py).

Run: python -m models.relabel_severity
Effect: data/synthetic/complaints.csv gets the new `severity`; the old label is kept in `severity_original`.
"""
import json
from pathlib import Path

import pandas as pd
from google.genai import types

from config import SEVERITY_LEVELS
from src.ai_service import _generate, _parse_json_response
import src.ai_service as ai_service

CSV = Path("data/synthetic/complaints.csv")
BATCH = 55

RELABEL_PROMPT = """You are labelling civic complaints from a college/hostel by SEVERITY.
Apply these definitions strictly and consistently — the same kind of problem must get the same label:

- Low: Minor inconvenience, no safety impact, few users affected (e.g. one flickering light, a wobbly chair, slow but working wifi, dusty corridor).
- Medium: Meaningful inconvenience, maintenance needed, not urgent (e.g. wifi down in one area, leaking tap, broken chair, bin not emptied, bus late).
- High: Significant impact on safety, accessibility, sanitation, or essential services (e.g. dark corridor/street at night, overflowing garbage or toilets, water leak flooding a floor, broken gate lock, no water supply).
- Critical: Immediate danger or serious hazard requiring urgent response (e.g. exposed live wires or sparks, fire, gas smell, sewage overflow, structural collapse, contaminated drinking water making people ill, blocked emergency access).

Judge the underlying problem, not the writer's tone: ignore exaggerations ("100 days", "dead body smell", "whole college") when deciding severity.

Return ONLY a JSON array of objects {{"id": <int>, "severity": "Low" | "Medium" | "High" | "Critical"}}, one per complaint, same ids.

COMPLAINTS:
{rows}
"""


def relabel(df: pd.DataFrame) -> dict[int, str]:
    labels = {}
    for start in range(0, len(df), BATCH):
        chunk = df.iloc[start:start + BATCH]
        rows = "\n".join(f'{i}: {t}' for i, t in zip(chunk.index, chunk.raw_complaint))
        for attempt in range(2):
            try:
                response = _generate(contents=[RELABEL_PROMPT.format(rows=rows)],
                                     config=types.GenerateContentConfig(temperature=0.0))
                for item in _parse_json_response(response.text):
                    if item.get("severity") in SEVERITY_LEVELS:
                        labels[int(item["id"])] = item["severity"]
                print(f"  rows {start}-{start + len(chunk) - 1}: done via {ai_service.last_model_used}")
                break
            except Exception as e:
                print(f"  rows {start}-{start + len(chunk) - 1}: attempt {attempt + 1} failed: {str(e)[:120]}")
    return labels


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    if "severity_original" not in df.columns:
        df["severity_original"] = df["severity"]
    labels = relabel(df)
    missing = [i for i in df.index if i not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} rows got no label (e.g. {missing[:5]}); CSV not modified. Re-run later.")
    df["severity"] = [labels[i] for i in df.index]
    df.to_csv(CSV, index=False)
    changed = (df.severity != df.severity_original).mean()
    print(f"\nRe-labelled {len(df)} rows; {changed:.0%} changed.")
    print(pd.crosstab(df.severity_original, df.severity, margins=True))
