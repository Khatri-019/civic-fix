"""
Dataset Step 7 (DATASET_PLAN.md section 10): validate and export the final train/test CSVs + a dataset card.

Step 4 (text review) was skipped by the lead's decision, so nothing here is marked human_verified:
  photo_label_inherited - text written by Gemini, labels from the team-verified photo
  seed_label_unchecked  - text-only row, labels from its generation seed, checked by nobody
A few known-bad contradicting texts are rewritten (REWRITES); recorded texts are never changed.

Run:    python -m tools.finalize_dataset
Input:  manifests/complaints_draft.csv, images.csv, audio.csv, recording_lines.csv
Output: data/real/train.csv, data/real/test.csv, manifests/complaints_reviewed.csv, data/real/DATASET_CARD.md
No Gemini calls.
"""
import sys
from pathlib import Path

import pandas as pd

from config import CATEGORIES, SEVERITY_LEVELS
from tools.build_image_manifest import MANIFESTS, OWNERS, ROOT, load_lead_flags
from tools.process_audio import SETUPS, corpus_wer

DRAFT = MANIFESTS / "complaints_draft.csv"
REVIEWED = MANIFESTS / "complaints_reviewed.csv"
TRAIN = ROOT / "train.csv"
TEST = ROOT / "test.csv"
CARD = ROOT / "DATASET_CARD.md"

SWAP_GUIDE_COLUMNS = ["raw_complaint", "category", "severity", "has_contradiction"]
EXTRA_COLUMNS = ["record_id", "image_id", "owner", "variant", "label_source", "text_source"]
MIN_PER_CATEGORY, MIN_PER_SEVERITY, MIN_CONTRADICTING, MAX_CATEGORY_SHARE = 20, 40, 60, 0.15

# Contradicting texts that slipped past Step 3's checks, rewritten by Claude from the photo (record_id -> text, reason)
REWRITES = {
    "tr_005_contradicting": (
        "a car fell into a huge crater in the middle of the road outside the gate and the whole lane is shut",
        "The photo shows an open drain pit on the footpath beside the road; the road itself is intact and open."),
    "sn_007_contradicting": (
        "the sewage pipe has burst and dirty water is flooding into all the houses on our street",
        "The photo shows hoses discharging into an open stream in a wooded area, with no houses or flooding."),
    "wm_001_contradicting": (
        "truckloads of rotting garbage have been dumped here and the pile is taller than the boundary wall",
        "The photo shows a handful of small plastic cups on bare soil, not a large dumped pile."),
    "ch_004_contradicting": (
        "thick black mould covers the entire wall from floor to ceiling and the room is completely unusable",
        "The photo shows peeling paint with damp patches on part of an outside wall, not mould over a whole room."),
}


def build_reviewed(draft: pd.DataFrame) -> pd.DataFrame:
    df = draft.copy()
    df["text_source"] = "gemini"
    for record_id, (text, reason) in REWRITES.items():
        mask = df.record_id == record_id
        df.loc[mask, ["raw_complaint", "contradiction_reason", "text_source"]] = [text, reason, "claude_rewrite"]
    df["label_source"] = df.image_id.map(lambda i: "photo_label_inherited" if i else "seed_label_unchecked")
    return df


def attach_media(df: pd.DataFrame, images: pd.DataFrame, audio: pd.DataFrame) -> pd.DataFrame:
    """Project-relative paths, so eval code can open them directly."""
    files = images.set_index("image_id").file
    df["image_file"] = df.image_id.map(lambda i: (ROOT / files[i]).as_posix() if i else "")
    clips = audio.groupby("record_id").file.apply(lambda s: ";".join((ROOT / f).as_posix() for f in sorted(s)))
    df["audio_files"] = df.record_id.map(clips).fillna("")
    return df


def validate(df: pd.DataFrame, images: pd.DataFrame, audio: pd.DataFrame, lines: pd.DataFrame) -> list[str]:
    problems = []
    for column in ("image_file", "audio_files"):
        paths = [p for value in df[column] if value for p in value.split(";")]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            problems.append(f"{len(missing)} {column} paths don't exist, e.g. {missing[:3]}")
    duplicated = df[df.raw_complaint.str.lower().str.strip().duplicated(keep=False)].record_id.tolist()
    if duplicated:
        problems.append(f"duplicate texts: {duplicated}")
    if df.record_id.duplicated().any():
        problems.append("duplicate record_ids")
    if not df.category.isin(CATEGORIES).all() or not df.severity.isin(SEVERITY_LEVELS).all():
        problems.append("category or severity values that aren't in config.py")

    small_categories = df.category.value_counts().loc[lambda s: s < MIN_PER_CATEGORY].to_dict()
    small_severities = df.severity.value_counts().loc[lambda s: s < MIN_PER_SEVERITY].to_dict()
    if small_categories or small_severities:
        problems.append(f"below the plan's minimums: {small_categories} {small_severities}")
    if df.has_contradiction.sum() < MIN_CONTRADICTING:
        problems.append(f"only {df.has_contradiction.sum()} contradicting rows (plan: >= {MIN_CONTRADICTING})")
    if df.category.value_counts(normalize=True).max() > MAX_CATEGORY_SHARE:
        problems.append("a category is above the plan's ~15% share")

    photos = df[df.image_id != ""]
    leaked = set(photos[photos.split == "train"].image_id) & set(photos[photos.split == "test"].image_id)
    if leaked:
        problems.append(f"photos in both splits: {sorted(leaked)}")
    forced = {i for i, f in load_lead_flags().items() if str(f.get("force_train", "")).lower() == "yes"}
    if set(photos[photos.split == "test"].image_id) & forced:
        problems.append("a force_train photo is in test")
    test = df[df.split == "test"]
    if set(test.category) != set(CATEGORIES) or set(test.severity) != set(SEVERITY_LEVELS):
        problems.append("the test split is missing a category or severity")
    # Replaces the plan's "every test row human_verified": Step 4 was skipped, so check the photo labels instead
    unverified = set(photos.image_id) - set(images[images.verified == "yes"].image_id)
    if unverified:
        problems.append(f"rows from unverified photos: {sorted(unverified)}")

    texts = df.set_index("record_id").raw_complaint
    changed = lines[lines.reference_text != lines.record_id.map(texts)].record_id.unique().tolist()
    if changed:
        problems.append(f"recorded texts changed or missing: {changed}")
    orphans = sorted(set(audio.record_id) - set(df.record_id))
    if orphans:
        problems.append(f"audio clips for unknown record_ids: {orphans}")
    return problems


def table(frame: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    frame = frame.reset_index()
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + rows)


def write_card(df: pd.DataFrame, images: pd.DataFrame, audio: pd.DataFrame) -> str:
    def crosstab(column: str) -> pd.DataFrame:
        frame = pd.crosstab(df[column], df.split, margins=True, margins_name="total")
        return frame.rename_axis(index=column, columns=None)

    wer_rows = [("all clips", audio)] + [(f"speaker {s}", audio[audio.speaker == s]) for s in OWNERS]
    wer_rows += [("shared lines (same 11 texts)", audio[audio.part == "shared"])]
    wer = pd.DataFrame([{"group": name, "clips": len(part), **{s: f"{corpus_wer(part, s):.1%}" for s in SETUPS}}
                        for name, part in wer_rows]).set_index("group")
    signed = [p.name for p in (ROOT / "consent").glob("*") if p.name != "README.md"]
    test_photos = df[(df.split == "test") & (df.image_id != "")].image_id.nunique()
    with_audio = df[(df.split == "test") & (df.audio_files != "")]

    card = f"""# CivicFix real dataset card

Generated by `tools/finalize_dataset.py`. English only.

## Contents
- **{len(df)} complaint texts**: {(df.image_id != "").sum()} from {df.image_id[df.image_id != ""].nunique()} photos (4 variants each) and {(df.image_id == "").sum()} text-only.
- **Train {(df.split == "train").sum()} / test {(df.split == "test").sum()} rows.** The split is by photo ({test_photos} test photos), so all texts of a photo are in the same split. Text-only rows are split by row.
- **{df.has_contradiction.sum()} contradicting rows**: a complaint that the photo visibly disproves (exaggeration, wrong fault or wrong scale).
- **{len(audio)} voice clips** ({audio.duration_s.sum() / 60:.1f} min) from 4 speakers (2 male, 2 female). {len(with_audio)} test rows have at least one recording.

## Label provenance
| label_source | rows | meaning |
|---|---|---|
| photo_label_inherited | {(df.label_source == "photo_label_inherited").sum()} | Gemini wrote the text; category and severity come from the photo, which the team verified |
| seed_label_unchecked | {(df.label_source == "seed_label_unchecked").sum()} | text-only row; labels are the generation seed, and no person checked them |

The texts were **not individually reviewed** (Step 4 was skipped), so there is no inter-annotator agreement score.
{len(REWRITES)} contradicting texts that failed a manual read were rewritten (`text_source = claude_rewrite`).
Contradicting rows keep the photo's severity, so their severity label doesn't describe their text. Leave them out
of severity training and severity scoring.

## Category
{table(crosstab("category"))}

## Severity
{table(crosstab("severity"))}

## Variant
{table(crosstab("variant"))}

## Owner (rows per member's categories)
{table(crosstab("owner"))}

## Speech recognition (word error rate, lower is better)
`app` = the app's Whisper setup (base, language auto-detected). `forced_en` = the same model forced to English.
`base_en` = Whisper's English-only model.

{table(wer)}

With 2 speakers per gender, speaker differences outweigh gender differences, so no gender effect is claimed.
Noisy clips used different texts from the quiet ones, so quiet vs noisy isn't a fair comparison.

## Images and consent
- Photo sources: {images.source.value_counts().to_dict()}. Licences: {images.license.value_counts().to_dict()}.
  Many photos are stock or web images (some watermarked), so **don't publish them**.
- Two photos (`in_006`, `in_009`) were cropped to remove faces. `sn_002` and `ch_003` show the same sink, so both are kept in train.
- Signed consent forms on file: {len(signed)} of 4.
"""
    CARD.write_text(card, encoding="utf-8")
    return card


def main() -> int:
    draft = pd.read_csv(DRAFT, keep_default_na=False, encoding="utf-8-sig")
    images = pd.read_csv(MANIFESTS / "images.csv", keep_default_na=False, encoding="utf-8-sig")
    audio = pd.read_csv(MANIFESTS / "audio.csv", keep_default_na=False, encoding="utf-8-sig")
    lines = pd.read_csv(MANIFESTS / "recording_lines.csv", keep_default_na=False, encoding="utf-8-sig")
    draft["has_contradiction"] = draft.has_contradiction.astype(str).str.lower() == "true"

    df = attach_media(build_reviewed(draft), images, audio)
    problems = validate(df, images, audio, lines)
    if problems:
        print("Validation failed, nothing exported:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    df.to_csv(REVIEWED, index=False, encoding="utf-8-sig")
    columns = SWAP_GUIDE_COLUMNS + EXTRA_COLUMNS
    df[df.split == "train"][columns].to_csv(TRAIN, index=False, encoding="utf-8")
    df[df.split == "test"][columns + ["image_file", "audio_files"]].to_csv(TEST, index=False, encoding="utf-8")
    write_card(df, images, audio)
    print(f"Validation passed. train.csv {(df.split == 'train').sum()} rows, test.csv {(df.split == 'test').sum()} rows, "
          f"{CARD.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
