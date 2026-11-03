"""
Dataset Step 2 (DATASET_PLAN.md section 5): image manifest + per-member review sheets.

Gemini looks at each photo once, WITHOUT seeing the team's labels, and gives a scene description, a proposed
severity and a category check. Humans stay in charge: `severity`, `verified`, `source`, `license` and `category`
are carried over from the existing review sheets, so re-running never overwrites a member's edits. Re-running
after members save their sheets is also how they get merged into manifests/images.csv.

Run:    python -m tools.build_image_manifest
Input:  data/real/images/<folder>/<prefix>_<NNN>.(jpg|png), SOURCES.txt, review/images_<ID>.csv, review/lead_flags.csv
Output: data/real/manifests/images.csv, data/real/review/images_<ID>.csv
Responses are cached per photo content, so re-runs and quota-interrupted runs don't repeat calls.
"""
import hashlib
import io
import itertools
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from config import CATEGORIES, SEVERITY_LEVELS
from src.ai_service import _generate, _parse_json_response
import src.ai_service as ai_service

ROOT = Path("data/real")
IMAGES = ROOT / "images"
REVIEW = ROOT / "review"
MANIFESTS = ROOT / "manifests"
CACHE = MANIFESTS / "cache" / "images"
ORIGINALS = REVIEW / "originals"
LEAD_FLAGS = REVIEW / "lead_flags.csv"

BATCH = 8           # photos per Gemini call: the free tier allows only ~20 calls/day per model
THROTTLE_SECONDS = 4
DUPLICATE_DISTANCE = 8  # out of 64 bits; catches re-saves and resizes, not crops (those go in lead_flags.csv)

PREFIXES = {  # file prefix -> (category, owner); DATASET_PLAN.md section 4.2
    "el": ("Electrical", "M1"),
    "se": ("Security", "M1"),
    "tr": ("Transport", "M1"),
    "in": ("Infrastructure", "M2"),
    "pw": ("Plumbing / Water", "M2"),
    "nw": ("Internet / Network", "M2"),
    "wm": ("Waste Management", "F1"),
    "sn": ("Sanitation", "F1"),
    "ot": ("Other", "F1"),
    "ch": ("Cleaning / Housekeeping", "F2"),
    "ff": ("Furniture / Facilities", "F2"),
}
OWNERS = ["M1", "M2", "F1", "F2"]
CATEGORY_ALIASES = {  # short names used in the team's first sheets
    "cleaning": "Cleaning / Housekeeping",
    "furniture": "Furniture / Facilities",
    "plumbing": "Plumbing / Water",
    "network": "Internet / Network",
}
SHEET_COLUMNS = ["image_id", "file", "category", "gemini_category_check", "scene_description", "severity_proposed",
                 "severity", "check_notes", "source", "license", "verified"]

PROMPT = """You are helping prepare a dataset of civic complaints from a college/hostel/neighbourhood.
You get {count} photos, each preceded by its IMAGE ID. Judge each photo independently and only from what is visible.

For each photo return:
- "image_id": the id given
- "scene_description": 1-2 neutral English sentences stating the visible problem and setting. Don't invent location or duration.
- "severity": exactly one of {severities}
    Low: Minor inconvenience, no safety impact, few users affected.
    Medium: Meaningful inconvenience, maintenance needed, not urgent.
    High: Significant impact on safety, accessibility, sanitation, or essential services.
    Critical: Immediate danger or serious hazard requiring urgent response.
- "category": exactly one of {categories}
- "has_clear_problem": false if no civic problem is recognisable without explanation
- "watermark_or_screenshot": true if there is a stock-photo watermark or a search/app screenshot overlay
- "identifiable_person_or_plate": true if a recognisable face or readable licence plate is visible

Return ONLY a JSON array with one object per photo."""


def read_sheet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:  # Excel's "CSV" save uses the Windows codepage
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="cp1252")


def scan_photos() -> list[dict]:
    photos, problems = [], []
    for path in sorted(p for p in IMAGES.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
        prefix = path.stem.split("_")[0]
        if not re.fullmatch(r"[a-z]{2}_\d{3}", path.stem) or prefix not in PREFIXES:
            problems.append(path.as_posix())
            continue
        category, owner = PREFIXES[prefix]
        photos.append({"image_id": path.stem, "path": path, "file": path.relative_to(ROOT).as_posix(),
                       "folder_category": category, "owner": owner})
    if problems:
        print(f"WARNING: {len(problems)} files skipped (expected <prefix>_<NNN>): {problems}")
    ids = [p["image_id"] for p in photos]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate image ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return photos


def load_sources() -> dict[str, tuple[str, str]]:
    sources = {}
    for txt in IMAGES.rglob("SOURCES.txt"):
        for line in txt.read_text(encoding="utf-8-sig").splitlines():
            parts = [s.strip() for s in line.split("|")]
            if len(parts) == 3:
                sources[Path(parts[0]).stem] = (parts[1], parts[2])
    return sources


def load_lead_flags() -> dict[str, dict]:
    if not LEAD_FLAGS.exists():
        return {}
    return read_sheet(LEAD_FLAGS).set_index("image_id").to_dict("index")


def load_existing_sheets() -> dict[str, dict]:
    """Rows of the current review sheets, first backing up each sheet as it was before this tool ever touched it."""
    rows = {}
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    for path in sorted(REVIEW.glob("images_*.csv")):
        if not (ORIGINALS / path.name).exists():
            shutil.copy2(path, ORIGINALS / path.name)
        for row in read_sheet(path).to_dict("records"):
            rows[row["image_id"]] = row
    return rows


def cache_path(photo: dict) -> Path:
    digest = hashlib.sha1(photo["path"].read_bytes()).hexdigest()[:12]
    return CACHE / f"{photo['image_id']}_{digest}.json"


def valid_response(item: dict) -> bool:
    return (item.get("severity") in SEVERITY_LEVELS and item.get("category") in CATEGORIES
            and bool(str(item.get("scene_description", "")).strip()))


def analyze_batch(batch: list[dict]) -> None:
    """One Gemini call; each valid answer is cached straight away so an interruption loses nothing."""
    parts = [PROMPT.format(count=len(batch), severities=SEVERITY_LEVELS, categories=CATEGORIES)]
    for photo in batch:
        buf = io.BytesIO()
        Image.open(photo["path"]).convert("RGB").save(buf, format="JPEG", quality=90)  # most are RGBA screenshots
        parts += [f"IMAGE ID: {photo['image_id']}",
                  types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")]
    response = _generate(contents=parts, config=types.GenerateContentConfig(
        temperature=0.0, response_mime_type="application/json"))
    answers = {item.get("image_id"): item for item in _parse_json_response(response.text)}
    for photo in batch:
        item = answers.get(photo["image_id"], {})
        if valid_response(item):
            item["model"] = ai_service.last_model_used
            cache_path(photo).write_text(json.dumps(item, indent=2), encoding="utf-8")


def analyze_missing(photos: list[dict], lead_flags: dict) -> bool:
    """Returns False if the run stopped early (quota/overload); a later re-run continues from the cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    held = [p["image_id"] for p in photos if is_held(p, lead_flags)]
    todo = [p for p in photos if not cache_path(p).exists() and p["image_id"] not in held]
    if held:
        print(f"Not sent to Gemini (privacy hold in {LEAD_FLAGS.name}): {held}")
    print(f"Gemini: {len(photos) - len(todo) - len(held)} cached, {len(todo)} to analyze in ~{-(-len(todo) // BATCH)} calls")

    for attempt in range(2):  # the second pass only retries photos whose answer was missing or invalid
        pending = [p for p in todo if not cache_path(p).exists()]
        # Sorted by content hash to mix categories: a batch of one category would hint at the expected answer
        pending.sort(key=lambda p: cache_path(p).name.split("_")[-1])
        for start in range(0, len(pending), BATCH):
            batch = pending[start:start + BATCH]
            try:
                analyze_batch(batch)
                print(f"  {batch[0]['image_id']}.. ({len(batch)} photos) via {ai_service.last_model_used}")
            except genai_errors.APIError as e:
                if e.code in (404, 429, 503, 504):  # every model in the fallback list failed
                    print(f"  Stopping: all Gemini models unavailable ({e.code}). Re-run later; progress is cached.")
                    return False
                print(f"  batch failed ({e.code}): {str(e)[:150]}")
            except (ValueError, AttributeError, TypeError) as e:  # malformed JSON or an empty response
                print(f"  batch failed: {str(e)[:150]}")
            time.sleep(THROTTLE_SECONDS)
    return all(cache_path(p).exists() for p in todo)


def is_held(photo: dict, lead_flags: dict) -> bool:
    return lead_flags.get(photo["image_id"], {}).get("hold", "").strip().lower() == "yes"


def normalize_category(value: str, fallback: str) -> str:
    value = value.strip()
    if value in CATEGORIES:
        return value
    return CATEGORY_ALIASES.get(value.lower(), fallback)


def normalize_severity(value: str) -> str:
    return value.strip().capitalize() if value.strip().capitalize() in SEVERITY_LEVELS else ""


def find_near_duplicates(photos: list[dict]) -> list[tuple[str, str]]:
    def dhash(path: Path) -> list[bool]:
        pixels = Image.open(path).convert("L").resize((9, 8)).tobytes()
        return [pixels[r * 9 + c] > pixels[r * 9 + c + 1] for r in range(8) for c in range(8)]

    hashes = {p["image_id"]: dhash(p["path"]) for p in photos}
    return [(a, b) for a, b in itertools.combinations(hashes, 2)
            if sum(x != y for x, y in zip(hashes[a], hashes[b])) <= DUPLICATE_DISTANCE]


def build_rows(photos: list[dict], existing: dict, sources: dict, lead_flags: dict, duplicates: list) -> pd.DataFrame:
    duplicate_of = {}
    for a, b in duplicates:
        duplicate_of.setdefault(a, []).append(b)
        duplicate_of.setdefault(b, []).append(a)

    rows = []
    for photo in photos:
        image_id = photo["image_id"]
        old = existing.get(image_id, {})
        path = cache_path(photo)
        gemini = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        category = normalize_category(old.get("category", ""), photo["folder_category"])
        proposed = gemini.get("severity", "")
        severity = normalize_severity(old.get("severity", "")) or proposed
        source, license_ = sources.get(image_id, ("", ""))
        source = old.get("source", "").strip() or source or "unknown"   # never assume "own" without evidence
        license_ = old.get("license", "").strip() or license_ or "unknown"

        notes = []
        if lead_flags.get(image_id, {}).get("note", "").strip():
            notes.append(lead_flags[image_id]["note"].strip())
        if gemini and gemini["category"] != category:
            notes.append(f"CATEGORY? Gemini says {gemini['category']}")
        if proposed and severity != proposed:
            gap = abs(SEVERITY_LEVELS.index(severity) - SEVERITY_LEVELS.index(proposed)) if severity else 0
            notes.append(f"SEVERITY{'??' if gap >= 2 else '?'} Gemini says {proposed}")
        if gemini.get("has_clear_problem") is False:
            notes.append("NO CLEAR PROBLEM?")
        if gemini.get("watermark_or_screenshot"):
            notes.append("WATERMARK/SCREENSHOT: find source + licence or replace")
        if gemini.get("identifiable_person_or_plate"):
            notes.append("FACE/PLATE? crop or blur")
        if image_id in duplicate_of:
            notes.append(f"NEAR-DUPLICATE of {', '.join(duplicate_of[image_id])}")

        rows.append({
            "image_id": image_id, "file": photo["file"], "owner": photo["owner"], "category": category,
            "gemini_category_check": gemini.get("category", ""),
            "scene_description": gemini.get("scene_description", ""),
            "severity_proposed": proposed, "severity": severity, "check_notes": "; ".join(notes),
            "source": source, "license": license_, "verified": old.get("verified", "").strip().lower(),
            "gemini_model": gemini.get("model", ""),
        })
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFESTS / "images.csv", index=False, encoding="utf-8-sig")  # BOM so Excel reads it as UTF-8
    for owner in OWNERS:
        df[df.owner == owner][SHEET_COLUMNS].to_csv(REVIEW / f"images_{owner}.csv", index=False, encoding="utf-8-sig")


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{len(df)} photos -> manifests/images.csv + review/images_<ID>.csv")
    print(f"\nSeverity (team labels):\n{pd.crosstab(df.category, df.severity, margins=True)}")

    print("\nPer member:  photos | High+Critical | analyzed | flagged | verified")
    for owner in OWNERS:
        part = df[df.owner == owner]
        serious = part.severity.isin(["High", "Critical"]).sum()
        short = "  <- plan wants >= 4" if serious < 4 else ""
        print(f"  {owner}: {len(part):3d} | {serious:2d} | {(part.scene_description != '').sum():3d} | "
              f"{(part.check_notes != '').sum():3d} | {(part.verified == 'yes').sum():3d}{short}")

    analyzed = df[df.severity_proposed != ""]
    if len(analyzed):
        print(f"\nTeam vs Gemini: severity agrees on {(analyzed.severity == analyzed.severity_proposed).mean():.0%}, "
              f"category on {(analyzed.category == analyzed.gemini_category_check).mean():.0%} of {len(analyzed)} photos")
    print(f"Source unknown: {(df.source == 'unknown').sum()} of {len(df)} photos")
    print(f"Not verified yet: {(df.verified != 'yes').sum()} (Step 3 refuses to run until this is 0)")


def merge_manifest(photos: list[dict]) -> pd.DataFrame:
    """Merge the current review sheets + cached answers into images.csv and the sheets. Makes no Gemini calls."""
    existing = load_existing_sheets()
    gone = sorted(set(existing) - {p["image_id"] for p in photos})
    if gone:
        print(f"WARNING: in the sheets but no matching photo, dropped from the output: {gone}")
    df = build_rows(photos, existing, load_sources(), load_lead_flags(), find_near_duplicates(photos))
    write_outputs(df)
    return df


def main() -> int:
    photos = scan_photos()
    if not photos:
        raise SystemExit(f"No photos found under {IMAGES}")

    complete = analyze_missing(photos, load_lead_flags())
    df = merge_manifest(photos)
    print_summary(df)
    if not complete:
        print("\nINCOMPLETE: some photos have no Gemini answer yet. Re-run this command to continue.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
