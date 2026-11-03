"""
Dataset Step 5 (DATASET_PLAN.md section 8): consent template + one recording script per speaker.

Each speaker gets 11 shared lines (one per category, all quiet, all test rows, so Whisper can be compared across
speakers and later evaluated on test rows) plus 13 lines of their own from other members' categories. No Gemini calls.

Run:    python -m tools.generate_recording_scripts
Input:  data/real/manifests/complaints_draft.csv
Output: data/real/scripts/<ID>.md, data/real/manifests/recording_lines.csv, data/real/consent/README.md
The result is frozen once created: people record from it, so a re-run refuses instead of changing the lines.
"""
import random
from pathlib import Path

import pandas as pd

from config import CATEGORIES, SEVERITY_LEVELS
from tools.build_image_manifest import MANIFESTS, OWNERS, PREFIXES, ROOT

DRAFT = MANIFESTS / "complaints_draft.csv"
LINES = MANIFESTS / "recording_lines.csv"
SCRIPTS = ROOT / "scripts"
CONSENT = ROOT / "consent" / "README.md"

OWN_LINES = 13
NOISY_PER_SPEAKER = 7   # of 24 lines, about the plan's 30%
MIN_WORDS = 10          # shorter lines don't reach the plan's 5 second minimum
MAX_TRIES = 20000
# Rows known to need rewriting in Step 7; recording them would leave audio that no longer matches its text
EXCLUDED_ROWS = {"tr_005_contradicting", "sn_007_contradicting", "wm_001_contradicting", "ch_004_contradicting"}

SCRIPT_TEMPLATE = """# Recording script: {speaker}

**24 lines, about 30 minutes.** Sign the consent form (`consent/README.md`) before recording.

## How to record
- Use any phone voice recorder and save as `.m4a`, `.mp3` or `.wav`. Make **one file per line**.
- Read naturally, as if complaining to a warden. Small slips are fine, so keep them. Don't try to pronounce typos or
  abbreviations literally: say what the person meant.
- Each clip should last 5-25 seconds.
- **quiet** = a quiet room. **noisy** = a corridor, a fan running, or outdoors.
- Name each file exactly as shown below (change the extension if yours isn't `.m4a`), and upload it to the shared
  `audio/{speaker}/` folder.

Some lines deliberately sound dramatic or don't match a real photo. Read them as written.

## Part A: shared lines (all 4 members record these)
{shared}

## Part B: your own lines
{own}

## Part C: optional, in your own words
Record 3-5 complaints without reading, about problems you have really seen. Name them
`{speaker}_<prefix>_<number>_free_quiet.m4a`, for example `{speaker}_el_003_free_quiet.m4a`, using a photo's id from
that category. Don't mention real people's names or room numbers.

## Done when
- [ ] Consent signed and uploaded to the shared `consent/` folder
- [ ] All Part A lines recorded
- [ ] At least 10 clips in total (24 recommended), named as shown
"""

CONSENT_TEXT = """# CivicFix voice recording consent

Each member signs one copy and uploads it to the restricted `consent/` folder. Signed forms are not committed to git.

**Speaker ID (M1 / M2 / F1 / F2):** ________

I agree to record my voice for the CivicFix project, on these terms:

1. **Purpose.** My recordings are used for speech-to-text testing and for results in the project report.
2. **Processing.** The audio is transcribed locally with Whisper on the project machine. The audio itself is not sent
   to Gemini. The text transcript may be sent to Gemini, as it is in the CivicFix app.
3. **Identification.** Files and results use my speaker ID only, not my name. Results may be reported by speaker ID and
   gender.
4. **Publication.** Raw audio will not be published without my permission.
5. **Deletion.** I can ask for my recordings to be deleted at any time, and they will be removed from the project.

**Name:** ____________________  **Signature:** ____________________  **Date:** ____________
"""


def load_eligible() -> pd.DataFrame:
    df = pd.read_csv(DRAFT, keep_default_na=False, encoding="utf-8-sig")
    df = df[~df.record_id.isin(EXCLUDED_ROWS) & (df.raw_complaint.str.split().str.len() >= MIN_WORDS)]
    return df.reset_index(drop=True)


def pick_shared(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """One test row per category, covering every severity, with some contradicting and text-only rows."""
    pool = df[df.split == "test"]
    for _ in range(MAX_TRIES):
        rows = pd.concat(pool[pool.category == c].sample(1, random_state=rng.randrange(2**31)) for c in CATEGORIES)
        if (set(rows.severity) == set(SEVERITY_LEVELS) and rows.has_contradiction.sum() == 2
                and (rows.image_id == "").sum() >= 1
                and rows.record_id.str.endswith("_informal").sum() >= 3
                and rows.record_id.str.endswith("_detailed").sum() >= 3):
            return rows
    raise SystemExit("No valid set of shared lines found; check the test rows.")


def pick_own(df: pd.DataFrame, owner: str, used: set, rng: random.Random) -> pd.DataFrame:
    """Lines from other members' categories, spread across categories and severities."""
    pool = df[(df.owner != owner) & ~df.record_id.isin(used)]
    for _ in range(MAX_TRIES):
        rows = pool.sample(OWN_LINES, random_state=rng.randrange(2**31))
        if (rows.category.value_counts().max() <= 2 and rows.category.nunique() >= 7
                and set(rows.severity) == set(SEVERITY_LEVELS) and 2 <= rows.has_contradiction.sum() <= 3
                and (rows.split == "test").sum() >= 3 and rows.record_id.str.endswith("_informal").sum() >= 3):
            return rows
    raise SystemExit(f"No valid set of lines found for {owner}.")


def build_lines(df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(42)
    shared = pick_shared(df, rng)
    used = set(shared.record_id)
    records = []
    for speaker in OWNERS:
        own = pick_own(df, speaker, used, rng)
        used |= set(own.record_id)
        noisy = set(rng.sample(list(own.record_id), NOISY_PER_SPEAKER))
        for part, rows in (("shared", shared), ("own", own)):
            for row in rows.to_dict("records"):
                env = "noisy" if row["record_id"] in noisy else "quiet"
                records.append({
                    "speaker": speaker, "gender": "male" if speaker.startswith("M") else "female", "part": part,
                    "record_id": row["record_id"], "env": env, "reference_text": row["raw_complaint"],
                    "category": row["category"], "severity": row["severity"], "split": row["split"],
                    "has_contradiction": row["has_contradiction"],
                    "filename": f"{speaker}_{row['record_id']}_{env}.m4a",
                })
    return pd.DataFrame(records)


def format_lines(rows: pd.DataFrame) -> str:
    blocks = []
    for number, row in enumerate(rows.sort_values("env").to_dict("records"), start=1):
        blocks.append(f"{number}. {row['reference_text']}  \n   `{row['filename']}` ({row['env']})")
    return "\n".join(blocks)


def write_outputs(lines: pd.DataFrame) -> None:
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    CONSENT.parent.mkdir(parents=True, exist_ok=True)
    lines.to_csv(LINES, index=False, encoding="utf-8-sig")
    CONSENT.write_text(CONSENT_TEXT, encoding="utf-8")
    for speaker in OWNERS:
        part = lines[lines.speaker == speaker]
        text = SCRIPT_TEMPLATE.format(speaker=speaker, shared=format_lines(part[part.part == "shared"]),
                                      own=format_lines(part[part.part == "own"]))
        (SCRIPTS / f"{speaker}.md").write_text(text, encoding="utf-8")


def print_summary(lines: pd.DataFrame) -> None:
    print(f"{len(lines)} lines -> {SCRIPTS.as_posix()}/<ID>.md + {LINES.as_posix()}")
    print("Speaker: lines | categories | severities | contradicting | test rows | noisy")
    for speaker, part in lines.groupby("speaker", sort=False):
        print(f"  {speaker}: {len(part)} | {part.category.nunique()} | {part.severity.nunique()} | "
              f"{part.has_contradiction.sum()} | {(part.split == 'test').sum()} | {(part.env == 'noisy').sum()}")
    print(f"Distinct texts to record: {lines.record_id.nunique()}")


def main() -> None:
    if LINES.exists():
        raise SystemExit(f"{LINES} already exists and people may have recorded from it. Delete it only if nobody has.")
    if set(PREFIXES[p][1] for p in PREFIXES) != set(OWNERS):
        raise SystemExit("Owner list doesn't match the prefix table.")
    lines = build_lines(load_eligible())
    write_outputs(lines)
    print_summary(lines)


if __name__ == "__main__":
    main()
