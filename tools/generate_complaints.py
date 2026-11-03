"""
Dataset Step 3 (DATASET_PLAN.md section 6): English complaint texts grounded in the verified photos.

Each photo gets 4 variants (informal, detailed, vague, contradicting). Hard-to-photograph problems also get
text-only rows from category + severity seeds. Labels come from the verified sheets, never from Gemini, and
photos are split into train/test as whole groups, so all texts of a photo stay together.

Run:    python -m tools.generate_complaints
Input:  data/real/review/images_<ID>.csv (every row verified), data/real/review/lead_flags.csv
Output: data/real/manifests/complaints_draft.csv, data/real/manifests/splits.json
Responses are cached, so re-runs and quota-interrupted runs don't repeat calls.
"""
import hashlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from config import CATEGORIES, SEVERITY_LEVELS
from src.ai_service import _generate, _parse_json_response
import src.ai_service as ai_service
from tools.build_image_manifest import MANIFESTS, PREFIXES, ROOT, load_lead_flags, merge_manifest, scan_photos

CACHE = MANIFESTS / "cache" / "complaints"
DRAFT = MANIFESTS / "complaints_draft.csv"
SPLITS = MANIFESTS / "splits.json"

PHOTO_BATCH = 4     # photos per Gemini call: the free tier allows only ~20 calls/day per model
SEED_BATCH = 10
AUDIT_BATCH = 100
THROTTLE_SECONDS = 4
MAX_ROUNDS = 3      # later rounds retry invalid answers and regenerate exact duplicates
TEST_SHARE = 0.2
MIN_WORDS, MAX_WORDS = 3, 40

VARIANTS = {  # variant -> record_id suffix
    "consistent_informal": "informal",
    "consistent_detailed": "detailed",
    "vague": "vague",
    "contradicting": "contradicting",
}
TEXT_ONLY_STYLES = ["consistent_informal", "vague", "consistent_detailed"]
TEXT_ONLY_SEEDS = {  # category -> {severity: rows}, for problems a photo can't show
    "Internet / Network": {"Low": 6, "Medium": 8, "High": 6},
    "Transport": {"Low": 4, "Medium": 8, "High": 5, "Critical": 3},
    "Security": {"Low": 1, "Medium": 2, "High": 4, "Critical": 3},
    "Other": {"Low": 3, "Medium": 3, "High": 3, "Critical": 1},
}
TEXT_ONLY_TOPICS = {
    "Internet / Network": "wifi outage or slowness, login portal failing, dead LAN port, no signal in a block",
    "Transport": "bus not arriving or late, cancelled route, overcrowding, unsafe driving, vehicle breakdown",
    "Security": "guard absent, theft, strangers entering at night, gate left open, alarm not working",
    "Other": "noise at night, stray animals, mosquitoes, canteen hygiene, anything fitting no other department",
}
HINDI_WORDS = re.compile(
    r"\b(hai|hain|nahi|nhi|kharab|bahut|paani|pani|bijli|kya|abhi|kal|aur|hua|gaya|bhai|yaar|din|ghante|hafte)\b", re.I)
# A contradicting row keeps the photo's labels, so a text saying all is well would train the classifiers on noise
NOT_A_COMPLAINT = re.compile(
    r"\b(brand new|newly|intact|spotless|neatly|perfect(ly)?|smoothly|functional|shining|no (damage|issues?|problems?)"
    r"|(is|are|was|were|working) (fine|clean|great)|working (fine|well|properly)|(just|freshly) (been )?"
    r"(cleaned|scrubbed|repaired|painted|installed))\b", re.I)

SEVERITY_RULES = """Low: Minor inconvenience, no safety impact, few users affected.
Medium: Meaningful inconvenience, maintenance needed, not urgent.
High: Significant impact on safety, accessibility, sanitation, or essential services.
Critical: Immediate danger or serious hazard requiring urgent response."""

SHARED_RULES = f"""Write the way real students and residents of a college, hostel or neighbourhood type complaints.
- English only. Informal Indian-English phrasing such as "since 3 days" is fine, but no Hindi or Hinglish words.
- Each text is {MIN_WORDS}-{MAX_WORDS} words. Vary the length, wording and opening; never start them all the same way.
- Each text must be unique: vary the details so that no two complaints in the dataset read the same."""

PHOTO_PROMPT = f"""You write realistic civic complaints, each grounded in a photo.
You get {{count}} photos. Each is preceded by its IMAGE ID, its verified category and severity, and a short scene note.
The labels are fixed: write texts that fit them, and treat the photo as the truth if the note differs.

{SHARED_RULES}

For each photo return these fields:
- "image_id": the id given
- "consistent_informal": short, casual: abbreviations, lowercase, small typos. Agrees with the photo.
- "consistent_detailed": fuller and well written. Agrees with the photo; adds a plausible location or duration.
- "vague": minimal information, naming the problem only loosely. Agrees with the photo.
- "contradicting": a sincere COMPLAINT about a problem in the same category as the photo, but with a claim that what is
  VISIBLE in this photo contradicts: the wrong kind of fault, a wrong scale, or an exaggeration the photo disproves.
  Example for a slowly dripping tap: "water is gushing out of the broken tap and the whole bathroom is flooded".
  It must report something wrong: never say things are fine, new, clean, intact or working, and never praise anyone.
  For a router showing a red light, BAD: "all the lights are green and the internet is fast" (nothing wrong).
  GOOD: "the router is burnt black and smoking, it has melted onto the shelf" (a much worse fault than the photo shows).
  Unverifiable details alone (a date, "since 3 days") do not count as a contradiction.
- "contradiction_reason": one sentence saying what in the photo contradicts that claim.

Severity definitions:
{SEVERITY_RULES}

Return ONLY a JSON array with one object per photo."""

SEED_PROMPT = f"""You write realistic civic complaints that come from experience rather than a photo.

{SHARED_RULES}

Styles:
- consistent_informal: short, casual: abbreviations, lowercase, small typos.
- consistent_detailed: fuller and well written, with a plausible location or duration.
- vague: minimal information, naming the problem only loosely.

Severity definitions:
{SEVERITY_RULES}

Each text must clearly fit its category and severity. Use different problems across rows of the same category.
Return ONLY a JSON array of objects {{{{"id": <id given>, "text": <complaint>}}}}, one per row.

ROWS:
{{rows}}"""


AUDIT_PROMPT = """Each line below is meant to be a complaint from a student or resident. Whether it is true doesn't matter.
List the ids of lines that do NOT report a problem: they say things are fine, clean, new, dry, intact, safe, secure,
organised or working normally, or they just describe something with nothing wrong.
Lines reporting a fault, damage, shortage, mess or danger, even a tiny or exaggerated one, are fine and must not be listed.
Return ONLY a JSON array of those ids, or [] if there are none.

{rows}"""


def clean_text(text) -> str:
    text = str(text or "").strip()
    for curly, plain in {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}.items():
        text = text.replace(curly, plain)
    return text


def valid_text(text: str) -> bool:
    return (MIN_WORDS <= len(text.split()) <= MAX_WORDS and text.isascii() and not HINDI_WORDS.search(text))


def check_ready(images: pd.DataFrame, lead_flags: dict) -> list[str]:
    """Reasons Step 3 must not run; an empty list means every photo is cleared."""
    problems = []
    unverified = images[images.verified != "yes"].image_id.tolist()
    if unverified:
        problems.append(f"{len(unverified)} photos not verified, e.g. {unverified[:5]}")
    held = [i for i, f in lead_flags.items() if str(f.get("hold", "")).strip().lower() == "yes"]
    if held:
        problems.append(f"photos on privacy hold in lead_flags.csv: {held}")
    no_answer = images[images.scene_description == ""].image_id.tolist()
    if no_answer:
        problems.append(f"no scene description yet, run tools.build_image_manifest first: {no_answer}")
    bad = images[~images.severity.isin(SEVERITY_LEVELS) | ~images.category.isin(CATEGORIES)].image_id.tolist()
    if bad:
        problems.append(f"invalid severity or category: {bad}")
    return problems


def build_seeds() -> list[dict]:
    seeds = []
    prefix_of = {category: prefix for prefix, (category, _) in PREFIXES.items()}
    owner_of = {category: owner for _, (category, owner) in PREFIXES.items()}
    for category, counts in TEXT_ONLY_SEEDS.items():
        number = 0
        for severity, count in counts.items():
            for _ in range(count):
                number += 1
                seeds.append({"seed_id": f"{prefix_of[category]}_t{number:02d}", "category": category,
                              "severity": severity, "owner": owner_of[category],
                              "style": TEXT_ONLY_STYLES[number % len(TEXT_ONLY_STYLES)]})
    return seeds


def assign_splits(images: pd.DataFrame, seeds: list[dict], lead_flags: dict) -> dict[str, str]:
    """About 20% of photos -> test, per category, picking the severity furthest below its share. Saved once,
    so a later label edit can't move a photo that reviewers have already treated as test."""
    if SPLITS.exists():
        return json.loads(SPLITS.read_text(encoding="utf-8"))

    rng = random.Random(42)
    forced = {i for i, f in lead_flags.items() if str(f.get("force_train", "")).strip().lower() == "yes"}
    wanted = {s: TEST_SHARE * n for s, n in images.severity.value_counts().items()}
    picked = {s: 0 for s in SEVERITY_LEVELS}
    splits = {image_id: "train" for image_id in images.image_id}

    for category, count in images.category.value_counts(ascending=True).items():
        pool = images[(images.category == category) & ~images.image_id.isin(forced)].to_dict("records")
        rng.shuffle(pool)
        for _ in range(min(max(1, round(TEST_SHARE * count)), len(pool))):
            choice = max(pool, key=lambda r: wanted[r["severity"]] - picked[r["severity"]])
            pool.remove(choice)
            picked[choice["severity"]] += 1
            splits[choice["image_id"]] = "test"

    for category in TEXT_ONLY_SEEDS:  # seeds run in severity order, so every 5th spreads across severities
        rows = [s for s in seeds if s["category"] == category]
        for position, seed in enumerate(rows, start=1):
            splits[seed["seed_id"]] = "test" if position % round(1 / TEST_SHARE) == 0 else "train"

    SPLITS.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def photo_cache(photo: dict) -> Path:
    digest = hashlib.sha1((ROOT / photo["file"]).read_bytes()).hexdigest()[:12]
    return CACHE / f"{photo['image_id']}_{digest}.json"


def seed_cache(seed: dict) -> Path:
    return CACHE / f"{seed['seed_id']}.json"


def ask_gemini(parts: list) -> list:
    response = _generate(contents=parts, config=types.GenerateContentConfig(
        temperature=0.8, response_mime_type="application/json"))
    return _parse_json_response(response.text)


def generate_photo_batch(batch: list[dict]) -> None:
    parts = [PHOTO_PROMPT.format(count=len(batch))]
    for photo in batch:
        buf = io.BytesIO()
        Image.open(ROOT / photo["file"]).convert("RGB").save(buf, format="JPEG", quality=90)
        parts += [f"IMAGE ID: {photo['image_id']} | category: {photo['category']} | severity: {photo['severity']} | "
                  f"scene note: {photo['scene_description']}",
                  types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")]
    answers = {item.get("image_id"): item for item in ask_gemini(parts)}
    for photo in batch:
        item = answers.get(photo["image_id"], {})
        texts = {variant: clean_text(item.get(variant)) for variant in VARIANTS}
        reason = clean_text(item.get("contradiction_reason"))
        if (reason and all(valid_text(t) for t in texts.values())
                and not NOT_A_COMPLAINT.search(texts["contradicting"])):
            data = {**texts, "contradiction_reason": reason, "model": ai_service.last_model_used}
            photo_cache(photo).write_text(json.dumps(data, indent=2), encoding="utf-8")


def generate_seed_batch(batch: list[dict]) -> None:
    rows = "\n".join(f"- id {s['seed_id']} | category: {s['category']} ({TEXT_ONLY_TOPICS[s['category']]}) | "
                     f"severity: {s['severity']} | style: {s['style']}" for s in batch)
    answers = {item.get("id"): item for item in ask_gemini([SEED_PROMPT.format(rows=rows)])}
    for seed in batch:
        text = clean_text(answers.get(seed["seed_id"], {}).get("text"))
        if valid_text(text):
            data = {"text": text, "model": ai_service.last_model_used}
            seed_cache(seed).write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_batches(pending: list, size: int, generate, name_key: str) -> bool:
    """Returns False if every model is unavailable, so the run should stop and be resumed later."""
    for start in range(0, len(pending), size):
        batch = pending[start:start + size]
        try:
            generate(batch)
            print(f"  {batch[0][name_key]}.. ({len(batch)}) via {ai_service.last_model_used}", flush=True)
        except genai_errors.APIError as e:
            if e.code in (404, 429, 503, 504):  # every model in the fallback list failed
                print(f"  Stopping: all Gemini models unavailable ({e.code}). Re-run later; progress is cached.")
                return False
            print(f"  batch failed ({e.code}): {str(e)[:150]}")
        except httpx.TransportError as e:  # a timeout isn't an APIError, so _generate doesn't fall back on it
            print(f"  Stopping: network problem ({type(e).__name__}). Re-run later; progress is cached.")
            return False
        except (ValueError, AttributeError, TypeError) as e:  # malformed JSON or an empty response
            print(f"  batch failed: {str(e)[:150]}")
            if len(batch) > 1 and not run_batches(batch, 1, generate, name_key):  # some batches fail every time
                return False
        time.sleep(THROTTLE_SECONDS)
    return True


def invalidate_duplicates(photos: list[dict], seeds: list[dict]) -> int:
    """Delete the cached answer of any item repeating an earlier text, so the next round rewrites it."""
    seen, removed = set(), 0
    entries = [(photo_cache(p), list(VARIANTS)) for p in photos] + [(seed_cache(s), ["text"]) for s in seeds]
    for path, fields in entries:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        texts = {data[f].lower() for f in fields}
        if texts & seen or len(texts) < len(fields):
            path.unlink()
            removed += 1
        else:
            seen |= texts
    return removed


def audit_contradicting(photos: list[dict]) -> int:
    """Delete the cached answer of any photo whose contradicting text doesn't report a problem, so it gets rewritten.
    Reading sentences is what a regex can't do: "the drain is completely dry with no clogging" has no fixed wording."""
    unchecked = {}
    for photo in photos:
        path = photo_cache(photo)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("complaint_checked"):
                unchecked[photo["image_id"]] = (path, data)
    removed = 0
    ids = list(unchecked)
    for start in range(0, len(ids), AUDIT_BATCH):
        chunk = ids[start:start + AUDIT_BATCH]
        rows = "\n".join(f"- {i}: {unchecked[i][1]['contradicting']}" for i in chunk)
        response = _generate(contents=[AUDIT_PROMPT.format(rows=rows)], config=types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json"))
        rejected = set(_parse_json_response(response.text))
        for image_id in chunk:
            path, data = unchecked[image_id]
            if image_id in rejected:
                print(f"    rejected {image_id}: {data['contradicting']}")
                path.unlink()
                removed += 1
            else:
                path.write_text(json.dumps({**data, "complaint_checked": True}, indent=2), encoding="utf-8")
        time.sleep(THROTTLE_SECONDS)
    return removed


def generate_all(photos: list[dict], seeds: list[dict]) -> bool:
    CACHE.mkdir(parents=True, exist_ok=True)
    for round_number in range(1, MAX_ROUNDS + 1):
        rng = random.Random(round_number)  # mix categories inside each batch
        pending_photos = [p for p in photos if not photo_cache(p).exists()]
        pending_seeds = [s for s in seeds if not seed_cache(s).exists()]
        rng.shuffle(pending_photos)
        rng.shuffle(pending_seeds)
        print(f"Round {round_number}: {len(pending_photos)} photos, {len(pending_seeds)} text-only rows to generate")
        if not run_batches(pending_photos, PHOTO_BATCH, generate_photo_batch, "image_id"):
            return False
        if not run_batches(pending_seeds, SEED_BATCH, generate_seed_batch, "seed_id"):
            return False
        if round_number < MAX_ROUNDS:
            removed = invalidate_duplicates(photos, seeds)
            if removed:
                print(f"  {removed} items repeated an earlier text; rewriting them")
        try:
            removed = audit_contradicting(photos)
        except (genai_errors.APIError, httpx.TransportError, ValueError, AttributeError, TypeError) as e:
            print(f"  Complaint check failed, re-run later; progress is cached: {str(e)[:150]}")
            return False
        print(f"  Complaint check: {removed} contradicting texts said nothing was wrong; rewriting them")
    return all(photo_cache(p).exists() for p in photos) and all(seed_cache(s).exists() for s in seeds)


def build_draft(photos: list[dict], seeds: list[dict], splits: dict) -> pd.DataFrame:
    rows = []
    for photo in photos:
        if not photo_cache(photo).exists():
            continue
        data = json.loads(photo_cache(photo).read_text(encoding="utf-8"))
        for variant, suffix in VARIANTS.items():
            is_contradiction = variant == "contradicting"
            rows.append({
                "record_id": f"{photo['image_id']}_{suffix}", "image_id": photo["image_id"], "owner": photo["owner"],
                "variant": variant, "raw_complaint": data[variant], "category": photo["category"],
                "severity": photo["severity"], "has_contradiction": is_contradiction,
                "split": splits[photo["image_id"]], "label_source": "gemini_draft",
                "contradiction_reason": data["contradiction_reason"] if is_contradiction else "",
                "gemini_model": data["model"],
            })
    for seed in seeds:
        if not seed_cache(seed).exists():
            continue
        data = json.loads(seed_cache(seed).read_text(encoding="utf-8"))
        rows.append({
            "record_id": f"{seed['seed_id']}_{VARIANTS[seed['style']]}", "image_id": "", "owner": seed["owner"],
            "variant": seed["style"], "raw_complaint": data["text"], "category": seed["category"],
            "severity": seed["severity"], "has_contradiction": False, "split": splits[seed["seed_id"]],
            "label_source": "gemini_draft", "contradiction_reason": "", "gemini_model": data["model"],
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    words = df.raw_complaint.str.split().str.len()
    print(f"\n{len(df)} rows -> {DRAFT.as_posix()}")
    print(f"From photos: {(df.image_id != '').sum()} | text-only: {(df.image_id == '').sum()} | "
          f"contradicting: {df.has_contradiction.sum()} | words: {words.min()}-{words.max()} (median {words.median():.0f})")
    print(f"\nCategory x split:\n{pd.crosstab(df.category, df.split, margins=True)}")
    print(f"\nSeverity x split:\n{pd.crosstab(df.severity, df.split, margins=True)}")
    print(f"\nRows per owner (test rows): "
          + ", ".join(f"{o}: {len(g)} ({(g.split == 'test').sum()})" for o, g in df.groupby("owner")))

    both = set(df[df.split == "train"].image_id) & set(df[df.split == "test"].image_id) - {""}
    duplicates = df.raw_complaint.str.lower().duplicated().sum()
    print(f"\nPhotos in both splits: {len(both)} | duplicate texts: {duplicates} | models: "
          f"{df.gemini_model.value_counts().to_dict()}")
    short_cat = df.category.value_counts().loc[lambda s: s < 20].to_dict()
    short_sev = df.severity.value_counts().loc[lambda s: s < 40].to_dict()
    top_share = df.category.value_counts(normalize=True).max()
    print(f"Plan minimums: categories under 20 rows: {short_cat or 'none'} | severities under 40 rows: "
          f"{short_sev or 'none'} | largest category: {top_share:.0%} (limit ~15%) | "
          f"contradicting >= 60: {df.has_contradiction.sum() >= 60}")


def main() -> int:
    images = merge_manifest(scan_photos())
    lead_flags = load_lead_flags()
    problems = check_ready(images, lead_flags)
    if problems:
        print("Step 3 refused, nothing was sent to Gemini:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    photos = images.to_dict("records")
    seeds = build_seeds()
    splits = assign_splits(images, seeds, lead_flags)
    complete = generate_all(photos, seeds)
    df = build_draft(photos, seeds, splits)
    if len(df):
        df.to_csv(DRAFT, index=False, encoding="utf-8-sig")  # BOM so Excel reads it as UTF-8
        print_summary(df)
    if not complete:
        print("\nINCOMPLETE: some items have no valid text yet. Re-run this command to continue.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
