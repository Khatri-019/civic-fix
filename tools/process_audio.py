"""
Dataset Step 6 (DATASET_PLAN.md section 9): transcribe the team's recordings locally with Whisper and score WER.

Three setups are compared on the same clips, to decide whether the app should force English:
  app       - exactly what the app does: src.speech_service.transcribe_audio (Whisper base, language auto-detected)
  forced_en - the same base model, told the speech is English
  base_en   - Whisper's English-only base.en model
WER is word-level edit distance against the script line, after expanding chat abbreviations in the script
("r" -> "are"; speakers were told to say what was meant) and applying Whisper's English normalizer to both sides.

Run:    python -m tools.process_audio
Input:  data/real/audio/<ID>/<ID>_<record_id>_<quiet|noisy>.<ext>, data/real/manifests/recording_lines.csv
Output: data/real/manifests/audio.csv
Transcripts are cached per setup and file content, so re-runs only transcribe new or changed clips.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import whisper
from whisper.normalizers import EnglishTextNormalizer

from src import speech_service
from tools.build_image_manifest import MANIFESTS, OWNERS, ROOT

AUDIO = ROOT / "audio"
LINES = MANIFESTS / "recording_lines.csv"
OUTPUT = MANIFESTS / "audio.csv"
CACHE = MANIFESTS / "cache" / "audio"
EXTENSIONS = (".m4a", ".mp3", ".wav", ".mp4", ".ogg", ".aac")
SETUPS = ["app", "forced_en", "base_en"]
FILENAME = re.compile(r"^(M1|M2|F1|F2)_(.+)_(quiet|noisy)$")

ABBREVIATIONS = {
    "r": "are", "n": "and", "u": "you", "ur": "your", "pls": "please", "plz": "please", "plss": "please",
    "v": "very", "rm": "room", "coz": "because", "cuz": "because", "b4": "before", "thx": "thanks", "w/": "with",
}
SPELLINGS = {  # applied to both sides after normalizing, so only spelling, never sound, is forgiven
    "as soon as possible": "asap", "wi fi": "wifi", "cant": "can not", "dont": "do not", "wont": "will not",
    "switch board": "switchboard", "sign post": "signpost", "door knob": "doorknob", "wash room": "washroom",
    "class room": "classroom", "foot path": "footpath", "dust bin": "dustbin",
}
normalizer = EnglishTextNormalizer()
_models = {}


def normalize(text: str, expand: bool) -> list[str]:
    if expand:
        text = text.replace("&", " and ")
        text = " ".join(ABBREVIATIONS.get(w.lower(), w) for w in text.split())
    text = re.sub(r"(\d)[:.](\d)", r"\1 \2", text)  # "8:30" and "8.30" are both said "eight thirty"
    text = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", text, flags=re.I)  # "a.m." -> "am"
    text = " " + normalizer(text) + " "
    for variant, canonical in SPELLINGS.items():  # same words written differently aren't recognition errors
        text = text.replace(f" {variant} ", f" {canonical} ")
    return text.split()


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over words: substitutions + deletions + insertions."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1]


def scan_clips(lines: pd.DataFrame) -> list[dict]:
    references = {(r.speaker, r.record_id): r for r in lines.itertuples()}
    clips, problems = [], []
    for path in sorted(p for p in AUDIO.rglob("*") if p.suffix.lower() in EXTENSIONS):
        match = FILENAME.match(path.stem)
        if not match or match[1] != path.parent.name:
            problems.append(path.as_posix())
            continue
        speaker, record_id, env = match.groups()
        line = references.get((speaker, record_id))
        clips.append({
            "audio_id": path.stem, "path": path, "file": path.relative_to(ROOT).as_posix(), "speaker": speaker,
            "gender": "male" if speaker.startswith("M") else "female", "record_id": record_id, "env": env,
            "part": line.part if line else "", "split": line.split if line else "",
            "reference_text": line.reference_text if line else "",
        })
    if problems:
        print(f"WARNING: {len(problems)} files skipped (expected audio/<ID>/<ID>_<record_id>_<env>.<ext>): {problems}")
    unknown = [c["audio_id"] for c in clips if not c["reference_text"]]
    if unknown:
        print(f"WARNING: no script line for {unknown}; transcribed but not scored")
    return clips


def transcribe(path: Path, setup: str) -> dict:
    if setup == "app":
        return speech_service.transcribe_audio(path)
    name, options = ("base", {"language": "en"}) if setup == "forced_en" else ("base.en", {})
    if name not in _models:
        _models[name] = whisper.load_model(name)
    result = _models[name].transcribe(str(path), fp16=False, **options)
    return {"success": True, "transcript": result["text"].strip(), "language": result.get("language", "en")}


def cached_transcribe(clip: dict, setup: str) -> dict:
    digest = hashlib.sha1(clip["path"].read_bytes()).hexdigest()[:12]
    path = CACHE / f"{setup}_{clip['audio_id']}_{digest}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = transcribe(clip["path"], setup)
    if result.get("success"):  # a failure is retried on the next run instead of being cached
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def process(clips: list[dict]) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    speech_service._ensure_ffmpeg()  # whisper.load_audio needs ffmpeg for the durations too
    rows = []
    for number, clip in enumerate(clips, start=1):
        row = {k: v for k, v in clip.items() if k != "path"}
        row["duration_s"] = round(len(whisper.load_audio(str(clip["path"]))) / whisper.audio.SAMPLE_RATE, 2)
        reference = normalize(clip["reference_text"], expand=True)
        row["reference_words"] = len(reference)
        for setup in SETUPS:
            result = cached_transcribe(clip, setup)
            prefix = "whisper" if setup == "app" else setup
            row[f"{prefix}_transcript"] = result.get("transcript") or ""
            row[f"{prefix}_language"] = result.get("language") or ""
            errors = word_errors(reference, normalize(row[f"{prefix}_transcript"], expand=False)) if reference else None
            row[f"{prefix}_errors"] = errors
            row["wer" if setup == "app" else f"{setup}_wer"] = round(errors / len(reference), 4) if reference else None
        rows.append(row)
        if number % 10 == 0 or number == len(clips):
            print(f"  {number}/{len(clips)} clips", flush=True)
    return pd.DataFrame(rows)


def corpus_wer(df: pd.DataFrame, setup: str) -> float:
    prefix = "whisper" if setup == "app" else setup
    scored = df[df.reference_words > 0]
    return scored[f"{prefix}_errors"].sum() / scored.reference_words.sum()


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{len(df)} clips -> {OUTPUT.as_posix()} | {df.duration_s.sum() / 60:.1f} min of audio")
    print("Corpus WER (total errors / total reference words), lower is better:")
    print(f"  {'group':<24}" + "".join(f"{s:>11}" for s in SETUPS) + "   clips")
    groups = [("all", df)] + [(f"speaker {s}", df[df.speaker == s]) for s in OWNERS]
    groups += [(f"{g}", df[df.gender == g]) for g in ("male", "female")]
    groups += [(f"{e}", df[df.env == e]) for e in ("quiet", "noisy")]
    groups += [("shared lines only", df[df.part == "shared"])]
    for variant in ("informal", "detailed", "vague", "contradicting"):
        part = df[df.record_id.str.endswith(f"_{variant}")]
        if len(part):
            groups.append((f"{variant} lines", part))
    for name, part in groups:
        print(f"  {name:<24}" + "".join(f"{corpus_wer(part, s):>10.1%} " for s in SETUPS) + f"  {len(part):>4}")
    wrong_language = df[df.whisper_language != "en"]
    print(f"\nApp setup detected a non-English language on {len(wrong_language)} clips"
          + (f": {wrong_language[['audio_id', 'whisper_language']].values.tolist()}" if len(wrong_language) else ""))
    print("Worst clips (app setup):")
    worst = df.sort_values("wer", ascending=False).head(5)
    for row in worst.itertuples():
        print(f"  {row.wer:.0%} {row.audio_id}\n      ref: {row.reference_text}\n      got: {row.whisper_transcript}")


def main() -> int:
    lines = pd.read_csv(LINES, keep_default_na=False, encoding="utf-8-sig")
    clips = scan_clips(lines)
    if not clips:
        raise SystemExit(f"No recordings found under {AUDIO}")
    df = process(clips)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print_summary(df)
    failed = (df.whisper_transcript == "").sum()
    if failed:
        print(f"\nINCOMPLETE: {failed} clips have no app transcript. Re-run to retry them.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
