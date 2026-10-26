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

# timeout (ms) so an overloaded model fails the batch instead of hanging
client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=180_000))

GENERATION_PROMPT = """Generate {n} realistic civic complaint examples for a college/hostel environment.

For each example, provide:
1. raw_complaint: informal English text (1-2 sentences, may include typos, abbreviations, vague language). English only — no Hindi or Hinglish words.
2. category: one of {categories}
3. severity: one of {severity_levels}
4. has_contradiction: true if the complaint contains an exaggerated or unverifiable claim, false otherwise

Return ONLY a JSON array. No markdown.

Example entry:
{{
  "raw_complaint": "light not working outside hostel since 3 days",
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
