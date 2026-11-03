# CivicFix — Project Memory

This file is loaded automatically at the start of every Claude Code session. The user clears chat between phases, so this is the **only** carry-over context. Keep it accurate.

## Workflow rules
- The build plan is [CIVICFIX_BUILD_PLAN.md](CIVICFIX_BUILD_PLAN.md). Execute **one phase at a time**, only the phase the user names.
- Read only the requested phase's section (plus the shared top sections: overview, env vars, structure). Do not read ahead or implement later phases.
- After a phase passes its acceptance criteria: **update the Phase Log below**, then stop and report. Wait for approval.
- Code in the plan is the spec; copy it as written unless it breaks against installed library versions — then fix minimally and record the deviation here.
- **Demo for the professor on Wed 2026-10-28.** Priority is a working end-to-end demo on **synthetic data**: Phase 6 runs as written in the build plan but **English only** (drop "Hinglish" from the generation prompt). The real dataset below is **deferred until after the demo**.
- **Later (post-demo): real dataset per [DATASET_PLAN.md](DATASET_PLAN.md)** (user decision, 2026-10-26): real photos + Gemini-written texts grounded in them + human-verified labels + 4-speaker voice recordings (M1, M2, F1, F2). **Dataset is English only (no Hinglish).** Work is split across a 4-member team (= the 4 speakers; M1 is the lead who runs Claude steps); each member owns specific categories (DATASET_PLAN.md §0), so tools must emit per-member review sheets in `data/real/review/` and merge them back. The user requests work as "Dataset Step N"; build only that step's `tools/*.py` script. The original synthetic generator is optional augmentation only. Phase 7 trains on `data/real/train.csv`; Phase 8 evaluates on `data/real/test.csv`.

## Environment
- OS: Windows 11. Shell: PowerShell (Git Bash also available).
- Project root = `C:\Users\khatr\Desktop\rag_project` (stands in for the plan's `civicfix/` folder; no subfolder).
- Python 3.13.7, venv at `.venv/`. Always use `.\.venv\Scripts\python.exe` (or activate first).
- Not a git repo (yet). `.env` exists with a working `GEMINI_API_KEY` (never print it) and `GEMINI_MODEL=gemini-3.5-flash`. `.env.example` must stay blank (the user once pasted the key there by mistake; it was moved to `.env`).
- **⚠️ Free-tier quota: 20 generate_content requests/day PER MODEL** (hit on 2026-10-26; resets ~18 h later, around midnight Pacific). A full app analysis = 3 calls, so only ~6 analyses/day/model; Phase 8 eval (15 cases × 3 calls = 45) exceeds one model's daily quota. Quotas are per model, so switching model (e.g. `$env:GEMINI_MODEL='gemini-3.1-flash-lite'`, which overrides `.env` because load_dotenv doesn't override) gets a fresh 20. The quota error names `gemini-3.6-flash` when calling `gemini-3.5-flash` (alias). Paid tier (billing enabled) removes this.
- Gemini models: `gemini-2.0-flash` and `gemini-2.5-flash` are retired (404). `gemini-3.8-flash` exists but was overloaded on 2026-10-26 (503/504). `gemini-3.5-flash` works (~8–12 s/call, uses thinking tokens). `gemini-3.1-flash-lite` is a fast (~2 s) fallback. List models: `client.models.list()`.
- Run app: `.\.venv\Scripts\python.exe -m streamlit run app.py`
- Headless check without a browser: `.\.venv\Scripts\python.exe -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('app.py').run(); print(at.exception)"`

### Installed versions (newer majors than plan minimums — watch for API changes)
streamlit 1.66.1 · google-genai 2.33.0 · pydantic 2.14.1 · torch 2.14.1 (CPU) · transformers 5.20.2 · sentence-transformers 6.2.0 · datasets 5.1.0 · evaluate 0.4.6 · openai-whisper 20250625 · pandas 3.1.0 · numpy 2.5.4 · scikit-learn 1.9.1

Also installed: imageio-ffmpeg 0.6.0 (bundled ffmpeg 7.1; added in Phase 2). No system ffmpeg on this machine.

Likely friction points: Phase 7 `TrainingArguments`/`Trainer` and `CrossEncoder` APIs under transformers 5 / sentence-transformers 6.

## Phase Log

| Phase | Status | Date |
|---|---|---|
| 0 — Project Setup | ✅ Done | 2026-10-26 |
| 1 — Streamlit UI (No AI) | ✅ Done | 2026-10-26 |
| 2 — Whisper STT | ✅ Done | 2026-10-26 |
| 3 — Pydantic Schemas | ✅ Done | 2026-10-26 |
| 4 — Gemini Analysis | ✅ Done | 2026-10-26 |
| 5 — Quality Score | ✅ Done | 2026-10-26 |
| 6 — Synthetic Data (English only, for the Wed demo) | ✅ Done | 2026-10-26 |
| 6R — Real dataset (DATASET_PLAN.md) | ✅ Done (Steps 1–3, 5–7; Step 4 skipped) → `data/real/train.csv`, `test.csv` | 2026-11-03 |
| 7 — DistilBERT + MiniLM | ✅ Done on synthetic (2026-10-26); **retrained on real data** (test: cat 61%, sev 39% / macro-F1 0.36) | 2026-11-03 |
| 8 — Evaluation Suite | ✅ Done (15-case pilot); **real test set: core parts done**, modality ablation pending | 2026-11-03 |
| 9 — UI Polish + Demo | ✅ Done | 2026-10-27 |
| 10 — Final Testing | — | |

### Phase 0 — Project Setup ✅
- Created dirs: `src/`, `models/`, `eval/results/`, `data/synthetic/`, `data/real/`, `sample_data/` (empty ones hold `.gitkeep`).
- Files: `requirements.txt`, `.env.example`, `.gitignore`, `config.py`, `app.py` (minimal header + "Phase 0 complete" info box), empty `__init__.py` in `src/`, `models/`, `eval/` — all verbatim from plan.
- Verified: `pip check` clean; `import config` prints 11 categories; Streamlit server healthy (HTTP 200) and AppTest shows 0 exceptions.
- Deviations: none in code. Project lives in repo root instead of `civicfix/` subfolder.

### Phase 1 — Streamlit UI (No AI) ✅
- `src/utils.py` verbatim from plan.
- `app.py`: header, 2:3 columns (inputs | results), text_area + audio uploader (with `st.audio` preview) + image uploader (with `st.image`), Analyze button (disabled unless text non-blank or a file is uploaded), Reset button, 4 bordered placeholder cards (Category & Severity, Structured Complaint, Contradiction Check, Quality Score), `status_placeholder = st.empty()` in results column, footer disclaimer.
- State: `st.session_state["analysis"]` (currently just records which modalities were present; Analyze shows an info box, no AI). Reset uses `on_click=reset_all` → clears session_state and bumps `input_version`; widget keys are `text_input_{v}`, `audio_file_{v}`, `image_file_{v}` so uploaders actually clear.
- Deviation: `use_container_width=True` is deprecated in Streamlit 1.66 → use `width="stretch"` everywhere.
- Verified: AppTest (disabled-when-empty, enable on text, analyze, reset clears text+results, whitespace stays disabled), utils helpers on a real PNG, live server health OK. AppTest can't drive file_uploader — image/audio display only checked by code review, worth a manual browser check.

### Phase 2 — Whisper STT ✅
- `src/speech_service.py` = plan code + `_ensure_ffmpeg()` called at the start of `transcribe_audio` (inside the try): if `ffmpeg` isn't on PATH, copies the imageio-ffmpeg binary to `%TEMP%\civicfix_ffmpeg\ffmpeg.exe` once and prepends that dir to `PATH`.
- `app.py`: on Analyze with audio → `audio_file.seek(0)` → `save_temp_audio` → `transcribe_audio` under `st.spinner`, temp file deleted in `finally`. Result stored in `st.session_state["analysis"]["transcription"]` (the dict returned by `transcribe_audio`, or None if no audio). Shown in `st.expander("📝 Voice Transcript")` with detected language; if failed **or transcript empty** (silent audio) → `st.warning("Voice transcription failed — continuing with text/image.")`.
- Whisper `base` model (139 MB) downloaded to `~/.cache/whisper` on first use; first Analyze with audio is slow (model load).
- Deviations: `imageio-ffmpeg>=0.5.0` added to `requirements.txt` + the `_ensure_ffmpeg` helper (ffmpeg not installed on system). An empty transcript counts as failure in the UI.
- Verified (Windows TTS-generated speech wav, 3 s silent wav, 24-byte corrupt .wav): speech transcribed exactly (lang=en) → expander; silent → empty transcript → warning; corrupt → success=False, logged error → warning; temp file removed in all cases. Phase 1 AppTest still passes; server health OK. mp3/m4a/ogg not tested but go through the same ffmpeg path.

### Phase 3 — Pydantic Schemas ✅
- `src/schemas.py` verbatim from plan: `SeverityLevel` (str Enum: Low/Medium/High/Critical — **case-sensitive**, "high" is rejected), `ContradictionFlag`, `StructuredIssue` (confidence ∈ [0,1]), `ContradictionAudit`, `QualityScore` (all fields ∈ [0,100]), `CivicFixReport`.
- Not yet used by `app.py` (wired in Phase 4).
- Verified under pydantic 2.14.1: both plan import commands work; invalid severity ("Extreme", "high") and confidence 1.5 / -0.1 raise `ValidationError`; 0.0/1.0 accepted; QualityScore 101 rejected; a full `CivicFixReport` JSON round-trips (`model_dump_json` → `model_validate_json`).
- Deviations: none. Note for Phase 4: Gemini output must use exact-case severity strings, or normalize before validation.

### Phase 4 — Gemini Analysis ✅
- `src/prompts.py` verbatim. `src/ai_service.py` verbatim except (a) `image_pil.convert("RGB")` before JPEG save (all 3 `sample_data/*.png` are RGBA → plan code would crash), (b) client built with `http_options=types.HttpOptions(timeout=120_000)` so an overloaded model errors out instead of hanging the UI.
- Model default changed `gemini-2.0-flash` → `gemini-3.5-flash` in `config.py`, `.env`, `.env.example` (2.0 retired).
- `sample_data/` currently holds `broken_light.png`, `garbage.png`, `water_leak.png` (user-supplied; PNG not the plan's .jpg names).
- `app.py`: `run_pipeline()` runs inside `st.status` (steps: input received → speech transcribed → image analyzed → issue classified → audit complete → complaint generated); result dict in `st.session_state["analysis"]` with keys `has_text/has_audio/has_image/transcription/issue/audit/complaint/error` (issue/audit are Pydantic objects). `run_id` counter keys the complaint text_area so each new analysis refreshes it.
- Errors: `AnalysisError` for user-facing stops (no key in config, nothing to analyze); other exceptions → `friendly_error()` maps API-key / 429-quota / model-404 / 503 to plain messages, logs full trace server-side. `generate_complaint` has no retry (per plan) — failures surface via friendly_error.
- Results UI: problem title, category/subcategory, severity badge (HTML span: Low #2e7d32 green, Medium #fbc02d yellow, High #ef6c00 orange, Critical #c62828 red), department, severity reason, location/duration ("Not specified"), confidence, recommended action; 4 evidence expanders; audit metrics (verified = # supported flags / contradicted / unverifiable) + claim details; editable complaint `st.text_area` + `st.code(edited, language=None, wrap_lines=True)` for its built-in copy button; Quality Score placeholder card (Phase 5).
- Verified offline (mocked client): malformed JSON retried once; invalid severity twice → RuntimeError; RGBA PNGs → image/jpeg parts; audit parsing + fallback; AppTest no-key → friendly error; invalid-key error → friendly text, no raw API error; all 4 severities render with correct colors; edited complaint flows to copy box. Phase 1 AppTest still passes.
- Verified live (gemini-3.5-flash): text-only (Hinglish), image-only (water_leak.png), text+image (garbage.png), and mismatched text "burst pipe/flooded" + broken_light.png → 2 contradicted claims, audit_passed=False, complaint flags the mismatch. Full UI run via AppTest → "Analysis complete", metrics + complaint rendered. A real invalid key → friendly message, no exception. ~25 s per full analysis (3 sequential calls). User confirmed image upload + analysis works in the browser.
- Observations for later phases: Gemini often returns `category` as "Main / Sub" strings (e.g. "Electrical / Lighting", "Plumbing / Water / Water Leak / Flooding") rather than an exact `config.CATEGORIES` value, and `department` doesn't match `config.DEPARTMENTS`. Not normalized (plan is spec) — matters for Phase 8 classification accuracy. Text-only inputs make every claim "unverifiable" (no image).
- User verified in browser: image upload + analysis, and Reset clears uploads/results.

### Phase 5 — Quality Score ✅
- `src/scoring.py` verbatim (pure functions, no LLM). `issue.severity in ("High", "Critical")` works because `SeverityLevel` is a str Enum.
- `app.py`: `run_pipeline` computes `result["quality"]` = `calculate_quality_score(issue, audit, text, has_image, has_audio)` (raw_text = typed text only, not the transcript) and `result["unsupported"]` = `count_unsupported_sentences(complaint, audit.verified_facts)` after complaint generation (status step "✓ Quality scored"). UI card "⭐ Complaint Quality Score": two `st.metric`s (Before raw / After AI report, delta), 4 `st.progress` bars (Completeness/Specificity/Evidence/Actionability), "Unsupported sentences: x / y". Scores are of the original generated complaint, not user edits.
- Verified using saved Phase 4 live outputs (no API): deterministic over repeated runs; raw < after in all 4 demo cases (text-only 55→83.8, image-only 25→76.2, text+image 77→96.2, contradiction 62.5→68.8); manual example (3 sentences, 1 unsupported) → (1, 3); edge cases (no facts, empty complaint); AppTest shows 4 bars + metrics, no exception.
- Observations for Phase 8: `verified_facts` = supported claims + observed_evidence only, so text-only reports score every sentence unsupported (4/4) since text claims are all "unverifiable". The contradiction case gains only ~6 points. The word-overlap check doesn't strip punctuation ("days," ≠ "days"). Left as spec.

### Phase 6 — Synthetic Data ✅ (English only)
- `models/data_generator.py` = plan code with (a) prompt changed to English only (no Hinglish), example entry made English; (b) client `http_options=types.HttpOptions(timeout=180_000)`.
- Run history: the first run on `gemini-3.5-flash` gave 4/10 batches (one 503, then 429 daily quota) → 80 rows. The remaining 7 batches ran on `gemini-3.1-flash-lite` (via env override) → 140 rows. Merged with the plan's filters (valid category/severity, dropna, dedupe) → **`data/synthetic/complaints.csv`: 220 rows**, plus an extra `source_model` column (harmless; Phase 7 reads only `raw_complaint` + label column). Raw parts kept as `complaints_batch1.csv` / `complaints_batch2.csv`.
- Checks: all 11 categories (min Sanitation/Other 11, max Plumbing 30); all 4 severities (Medium 81, High 63, Low 53, Critical 23); 0 empty texts; 0 Hindi words; 54 `has_contradiction=True`; 8–21 words per row. Some category×severity cells are 0, which is fine because the two classifiers are trained separately.

### Gemini model fallback (added after Phase 6; user chose free tier, no billing)
- `config.GEMINI_FALLBACK_MODELS` (env `GEMINI_FALLBACK_MODELS`, comma list; default `gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.7-flash`). Don't put an empty `GEMINI_FALLBACK_MODELS=` line in `.env` (it would disable fallback).
- `src/ai_service._generate()` wraps every Gemini call: tries `GEMINI_MODEL` then the fallbacks. 429/404 → model skipped for the rest of the process (`_exhausted_models`); 503/504 → 10-min cooldown (`_cooldown_until`); other errors are raised. `ai_service.last_model_used` is shown in the UI ("Gemini model used: …").
- Model status on 2026-10-26: 3.5-flash and 3.6-flash share one quota (exhausted); 3.8-flash 503/504; 3.7-flash and flash-latest 504 after ~60 s; 3.5-flash-lite OK; 3.1-flash-lite intermittent 503.
- Live end-to-end before the cooldown existed took 300 s (repeated 504s on 3.8-flash) and was served by 3.5-flash-lite. With the cooldown, only the first 504 per 10 min costs ~60 s.

### Phase 7 — DistilBERT + MiniLM ✅
- `models/classifier.py` = plan code except: `logging_dir` removed from `TrainingArguments` (gone in transformers 5); `predict()` looks up `labels[str(label_id)]` (plan bug: JSON keys are strings → KeyError). Needed `accelerate>=1.1.0` (installed 1.15.0, added to requirements.txt).
- `models/contradiction_model.py`, `src/contradiction.py` verbatim. MiniLM `id2label` confirmed {0: contradiction, 1: entailment, 2: neutral}, matching the plan's map.
- First training (plan's 4 epochs): val category 0.59, severity 0.48 (Critical recall 0). User approved retraining "till a satisfiable threshold". Targets set beforehand: category ≥ 0.80, severity ≥ 0.65 with Critical recall > 0.
- Retraining changes: `--epochs` CLI arg (default now **15**); `train(..., epochs)`; `save_total_limit=1` + `seed=42` in TrainingArguments (each checkpoint is 766 MB; leftover `checkpoint-*` dirs were deleted, and the final models are the files at the dir root). 15 epochs → **category 0.80** val. Severity plateaued ~0.50 because the **synthetic severity labels were inconsistent** (TF-IDF+LR 5-fold CV also 0.49; e.g. "chair missing leg"=Low vs "chair loose"=Medium).
- Fix: new `models/relabel_severity.py` (`python -m models.relabel_severity`) re-labels every row's severity with one rubric (same definitions as src/prompts.py plus examples; "ignore exaggerations") via `ai_service._generate`, 4 calls of 55 rows. 48% of labels changed; the old label is kept in column `severity_original`; backup at `data/synthetic/complaints_before_relabel.csv`. Distribution after: Medium 78, High 63, Low 55, Critical 24. TF-IDF+LR CV rose to 0.62.
- Severity retrained (15 epochs, best = epoch 10) → **severity 0.73** val (Critical recall 0.40). Category model was trained before the relabel (category labels unchanged).
- Out-of-distribution probe (20 hand-written demo-style sentences): category 16/20, severity 13/20. Known misses: street light → Security, pothole → Security, overflowing toilet → Cleaning, ceiling fan → Infrastructure, "live wire, sparks" → **Low** (too few Critical hazards in training data). Next improvement = more data (generate ~300 more rows + relabel, ~15 Gemini calls) once quotas reset.
- `app.py`: `run_custom_models()` (lazy imports, never raises) runs after `analyze_issue`. DistilBERT input = typed text + transcript, or `issue.problem` for image-only. Stores `result["custom"]` = {custom_category, custom_severity, custom_input, nli_audit, custom_error}. UI: "🧪 Classification — Gemini vs Custom (DistilBERT)" card (✅ when Gemini's category starts with the custom label), and the contradiction card has tabs "Gemini audit (used for complaint)" / "MiniLM NLI audit (custom)" via `render_audit()`. The complaint still uses the Gemini audit.
- MiniLM behaviour: aggressive. "light broken 3 days" vs "streetlight fixture visible" → contradiction; on the saved text+image case all 6 claims → unverifiable; on the contradiction case it correctly flags the 2 false claims.
- Verified: model dirs exist; predict returns valid labels (incl. plan's "light kharab hai" → Electrical); check_contradiction returns dict; run_nli_audit works; AppTest (mocked Gemini) shows the comparison card + both tabs, no exception; one live run OK.

### Phase 7R — DistilBERT retrained on the real dataset ✅ (2026-11-03)
- **Backup:** the synthetic models were copied to `models/saved_synthetic/` (git-ignored, 513 MB). To bring them back for a demo: delete `models/saved/` and copy `models/saved_synthetic/` to `models/saved/`.
- `models/classifier.py` changes:
  - `load_data` returns `groups` (`image_id`, falling back to `record_id`). When present, validation uses `StratifiedGroupKFold` (20%, by photo). The plain random split would put a photo's other texts in validation and inflate the score. Synthetic data has no `image_id`, so it takes the old path.
  - `--severity-skip-contradicting`: drops `has_contradiction` rows from the severity model only (the agreed rule).
  - `--balanced-severity`: `WeightedTrainer` (class-weighted cross-entropy) + best epoch chosen by `macro_f1` (now in `compute_metrics`).
  - `--only category|severity`; `classification_report` lists all classes with `zero_division=0`.
- Commands used: `python -m models.classifier --train --data data/real/train.csv --only category` (equivalent; the category model came from the first full run) and `... --only severity --severity-skip-contradicting --balanced-severity`. 15 epochs, about 5 min per model on CPU. Leftover `checkpoint-*` dirs deleted.
- **Test results** (75 non-contradicting test rows; scored with a scratch script, Phase 8 will formalize):

  | | category acc | severity acc | severity macro-F1 |
  |---|---|---|---|
  | DistilBERT real | **61.3%** | 38.7% (balanced) | **0.357** |
  | DistilBERT synthetic | 49.3% | 24.0% | 0.182 |
  | TF-IDF + LR baseline | 56.0% | 34.7% | 0.278 |
  | majority class | – | 41.3% | 0.146 |

  - Unweighted severity training collapsed to always "Medium" (41.3% acc, 0 recall elsewhere), so the balanced model was kept.
  - The balanced model's recall: Low 81%, Medium 29%, High 26%, Critical 22%; it over-predicts Low.
- **Severity isn't learnable from text on this data.** No subset (variant, no vague rows, photo rows only) beats about 44% / 0.34 macro-F1 even for TF-IDF. Labels come from the photo (team's scale, 36% agreement with Gemini), while the model sees only text. Report this as a finding, not a tuning gap.
- Known demo misses with the real models: street light → **Transport** (was Security), "live wire … sparking" → **Low**. Garbage, water leak, pipe burst and wifi are correct. AppTest start: no exceptions.

### Phase 8 — Evaluation Suite ✅
- `eval/metrics.py` verbatim. `eval/eval_runner.py` = plan code plus these deviations:
  1. Test case 1 made English ("light not working outside hostel since 3 days"), per the English-only decision.
  2. `normalize_category()` maps Gemini's "Main / Sub" category to the config category it starts with (otherwise Gemini is scored wrong on correct answers).
  3. **Per-case Gemini cache** `eval/results/cache/<sha1(text)[:12]>.json` (issue, audit, complaint, complaint_no_audit, model_used). Re-runs and resumes cost 0 quota; delete the folder to re-query.
  4. **Hallucination metric fixed.** The plan scored the SAME complaint against audit.verified_facts vs all facts; verified ⊆ all, so "with audit ≤ without" was impossible by construction (got 0.89 vs 0.34). Now two complaints are generated (with the audit, and without it = every extracted fact passed as verified) and both are scored against the same grounded reference: raw text + user_reported_facts + observed_evidence. The plan's original numbers are kept in the report under `hallucination_reduction.plan_original_metric_same_complaint`.
  5. Report adds `evaluated_cases`, `failed_cases`; per-case details in `eval/results/eval_cases.csv` (incl. both complaints).
  6. `MODALITY_SETS` still defined but unused (as in the plan); the eval is text-only.
- Results (15 cases, all served by gemini-3.5-flash-lite via fallback; 60 Gemini calls total):
  - Category accuracy: **Gemini 100%** (after normalization) vs **DistilBERT 53%** (weighted F1 1.00 vs 0.48). DistilBERT misses: street light/road/debris → Security, toilet/sweeping → Plumbing, projector → Network, light → Other.
  - Severity accuracy: **Gemini 73%** vs **DistilBERT 67%** (weighted F1 0.71 vs 0.57).
  - Contradiction (3 positives, all exaggeration/unverifiable claims in text-only cases): **Gemini 0/3** (with no image its audit marks claims "unverifiable", never "contradicted"); **MiniLM P 0.50 / R 0.33** (1 TP, 1 FP on the sweeping case). This needs image evidence → motivates the real dataset.
  - Hallucination: unsupported-sentence rate **with audit 0.433 ≤ without 0.456** ✅ (small margin, n=15). Plan metric: 0.89 vs 0.34.
  - Avg quality score 73.7.
- Run notes: the first full run took ~18 min (fallback 503/504 waits); one resume was needed after a 504 on case 11. All acceptance criteria pass.

<!-- Template for next phases:
### Phase N — Name ✅
- What was built (files + key functions)
- How it was verified (commands / results)
- Deviations from plan + why
- Known issues / TODOs carried forward
-->

### Phase 8R — Evaluation on the real test set (core parts ✅ 2026-11-03)
- `eval/real_eval.py`; run with `python -m eval.real_eval [--part classification|contradiction]`. `eval_runner.py` (the 15-case pilot) is unchanged. Output: `eval/results/real_eval_report.json`, `real_eval_cases.csv`, per-case Gemini cache `eval/results/real_cache/{text,photo}_<record_id>.json`.
- Run on lite models only, to keep `gemini-3.5-flash` / `3.8-flash` quota for the demo: `$env:GEMINI_MODEL='gemini-3.5-flash-lite'; $env:GEMINI_FALLBACK_MODELS='gemini-3.1-flash-lite'`. 159 calls, 45 min (many 503s); 3.5-flash-lite hit 429 partway and 3.1-flash-lite finished.
- `gemini_with_photo` raises if the audit made no real Gemini call (`run_contradiction_audit` swallows errors and returns a "passed" fallback). Without this, quota errors would be cached as false negatives. Tested with a mocked 429.
- **Classification** (75 non-contradicting test rows, text only; accuracy with Wilson 95% CI):
  - category: Gemini 58.7% [47–69], macro-F1 0.62; DistilBERT 61.3% [50–72], 0.59.
  - severity: Gemini 36.0% [26–47], 0.35; DistilBERT 38.7% [28–50], 0.36.
  - The CIs overlap heavily: **no significant difference** at n=75.
  - 28 of Gemini's 31 category misses are "Infrastructure" for rows the team labelled Transport / Security / Other / Furniture / Network / Electrical (roads, walls, fallen trees, signposts). That's a **taxonomy-convention gap**: DistilBERT learned the team's conventions and Gemini didn't. It isn't Gemini misunderstanding the text.
  - Severity: Gemini rates higher than the team (17 of 31 Medium → High).
  - On which rows each is right: both wrong 13, only Gemini right 16, only DistilBERT right 18.
- **Contradiction** (21 contradicting + 21 matching texts on the same 21 test photos; Gemini = app's review trigger, i.e. contradicted claims or `image_mismatch`):
  - **Gemini P 1.00 / R 0.90 / F1 0.95, 0% false alarms on controls.**
  - **MiniLM P 0.78 / R 1.00 / F1 0.88, 29% false alarms.** MiniLM flags uncertain or extra claims ("may be held only by cables") as contradictions.
  - Gemini missed `nw_004` and `wm_009` (both contradicting). 2 of its catches came only through `image_mismatch`.
- `app.py`: the sidebar shows "Model Comparison (real test set)" from `REAL_EVAL_REPORT_PATH` (via `render_real_eval_summary`, which handles a partial report); the old 15-case table is in an "Earlier pilot" expander. AppTest: no exceptions, both tables render.
- **Pending (needs quota, later day):** modality ablation (text / photo / voice transcript / all on test photos, using `audio_files`) and the with/without-audit hallucination metric on real data.

### Phase 9 — UI Polish + Demo ✅
- `app.py` additions (all four plan items):
  1. **Before vs After** card under the issue card: original input (typed text + transcript, or "(image only…)", stored as `result["original_input"]`) vs `issue.problem (category / severity)`, with Raw / AI quality.
  2. **Sidebar Evaluation Summary** (`render_eval_summary()`), reading `eval/results/eval_report.json`: an `st.table` (Gemini vs Custom: category acc, severity acc, contradiction P/R) plus the hallucination rates. Accuracy comes from `per_class["accuracy"]` (fallback: weighted recall). If the report is missing → `st.info("Run eval suite first…")`.
  3. **⬇️ Download Report** (`st.download_button`, `build_text_report()`): the plan's plain-text format plus location, duration and recommended action. It uses the **edited** complaint text. File name `civicfix_report_<date>.txt`.
  4. **Sidebar Architecture** diagram (`st.code`, line-wrapped to fit the sidebar).
- Additional changes:
  - Placeholder/help text switched to English.
  - `run_custom_models` imports moved inside the try blocks, so a missing torch/model can't crash the app; a missing model dir already returns None → "not trained yet" caption, and the app stays Gemini-only.
- **Demo cache (beyond plan, for quota safety):**
  - `demo_cache/<sha1(text, transcript, image bytes)[:16]>.json` stores issue/audit/complaint/model_used. On a hit, the 3 Gemini calls are skipped. The custom models and scoring still run.
  - Sidebar toggle "Reuse cached results for identical inputs" (default on); the UI caption says "loaded from demo cache". The cache lookup happens after transcription and after the API-key check. `demo_cache/` and `eval/results/cache/` were added to `.gitignore`.
  - **Pre-warmed** for the 3 demo scenarios (wav + png, no typed text). Re-uploading those exact files makes the demo work even with 0 quota. Don't run AppTests with a mocked Gemini without cleaning the cache afterwards: they write fake entries.
- Sample data:
  - `sample_data/{broken_light,garbage,water_leak}.wav` generated with Windows TTS.
  - `sample_data/voice_samples.md` holds the 3 transcripts plus the contradiction demo (text "pipe burst…flooded" + `broken_light.png`, which is not cached).
  - The images stay `.png` (user-supplied), not the plan's `.jpg`.
- `README.md` follows the plan, plus a `relabel_severity` step and notes (synthetic data, quota/fallback, demo cache).
- Live verification (2026-10-27, gemini-3.5-flash-lite), voice + photo for each scenario:
  - broken_light → Gemini Electrical/High, DistilBERT **Security**/High (known miss), quality 45→96;
  - garbage → Waste Management/High on both, 45→94;
  - water_leak → Plumbing/High on both, 45→94.
- Offline AppTest checks (mocked Gemini):
  - eval table and architecture rendered; before/after panel and download button shown;
  - second identical run served from cache with Gemini failing (0 calls);
  - in an empty working dir (no models, no report): "Run eval suite first" plus "not trained yet", with complaint and download still working;
  - report text format checked.
- Regressions: the Phase 1 and Phase 7 AppTests pass. Phase 1 needed `default_timeout=120`, because the 3 s default is too short now. Server health 200.

### Post-Phase 9 change — input mismatch review (user request, 2026-10-27)
- User's flow: if text, voice and photo all agree → generate the report. If the photo contradicts them → warn the user, ask them to verify or re-upload, and offer to continue using the **text as the most trusted source**, with the report flagged. Video was dropped: the user had added it by mistake, and the app has no video input.
- `app.py`:
  - **Trigger:** `needs_user_review()` fires when an image was uploaded AND the Gemini audit has `contradicted_claims`. Text-only inputs can't trigger it, because their claims are only ever "unverifiable".
  - **Pipeline split:** `run_pipeline` stops after the audit (complaint=None, `needs_review=True`; status "Please verify your inputs"), so no complaint call is spent before the user decides. `finish_report(result, mode)` generates and scores the complaint: mode `"verified"` (the normal path) or `"text_priority"`.
  - **Review card:**
    - shows the contradicted claims with their reasons, plus a "What the photo actually shows" expander;
    - 📷 **Re-upload photo** clears only the image uploader (separate `image_version` key suffix) and the results, then shows a hint;
    - ✏️ **Edit description** clears the results and shows a hint;
    - ➡️ **Continue with my description** runs `finish_report(..., "text_priority")`.
  - **text_priority mode:**
    - **re-runs `analyze_issue(text, transcript, None)` without the image**, because the combined analysis takes the problem/category from the wrong photo (live: burst-pipe text + light photo → "Electrical"; after Continue → Plumbing / Water, Water and Sanitation dept). The combined issue is kept as `result["issue_original"]`.
    - The complaint is written via the new `ai_service.generate_text_priority_complaint()` and `prompts.TEXT_PRIORITY_COMPLAINT_PROMPT`: user-reported facts only, observed/inferred evidence excluded, and a final sentence saying the photo could not confirm the details.
    - UI: a "⚠️ Flagged" warning plus "What the photo shows (not used)". The downloaded report gets a `FLAG:` line. Unsupported sentences are measured against verified facts + user-reported facts.
    - Cost: Continue = 2 Gemini calls (re-analysis + complaint). A matched input still costs 3 calls in total.
  - **Demo cache:** the format is now merged keys: `issue`, `audit`, `model_used`, `complaint` (verified), plus `issue_text_priority` and `complaint_text_priority`. Old entries are compatible.
- Pre-cached: the contradiction demo (typed text "A pipe burst and the whole street is flooded, water everywhere for a week." + `broken_light.png`), including the Continue path. Verified that the app's upload path (save_temp_audio → Whisper) produces the same cache keys as the pre-warm script. `demo_cache/` now holds 6 entries: 3 demos + contradiction + 2 from the user's own browser runs.
- Tests:
  - **Offline (mocked):** review triggered, with no complaint call before the decision; Continue re-analyzes without the image; cached Continue makes 0 Gemini calls; matching inputs skip review; the report has the FLAG line.
  - **AppTest (preset session state):** card, 3 buttons, re-upload/edit clear the results and show hints (`image_version` bumped), Continue → flagged complaint and download.
  - **Live:** run once (as above).
  - **Regressions:** Phase 1, 7 and 9 tests pass. Mocked test runs write fake `demo_cache` entries; they were removed afterwards, so always clean the cache after mocked AppTests.
- **Fix 2026-10-27 (user report: "report still generated after wrong image"):** the user's wrong photo (notebook pages + water-leak voice) left every claim "unverifiable", so nothing was "contradicted" and no review fired.
  - **Fix:** `CONTRADICTION_PROMPT` now also returns `image_matches_report` + `mismatch_reason`. `ContradictionAudit` gained `image_mismatch: bool = False` and `mismatch_reason: Optional[str] = None` (defaults, so old cache/eval JSON still loads). `run_contradiction_audit` sets `image_mismatch` only when there is observed evidence AND the field is explicitly false; `audit_passed` is then false. `needs_user_review` = has_image and (contradicted claims OR image_mismatch). The review card shows the reason; the report FLAG line says "does not match".
  - **Live matrix (gemini-3.5-flash-lite):** 3 matching demo pairs → no review (no false alarms); leak voice + garbage photo, garbage voice + light photo, contradiction text + light photo → review with a clear reason. The demo cache was rebuilt from scratch with the new audits (6 entries: 3 demos, 2 wrong-photo voice pairs, contradiction incl. Continue path).
- **Terminal noise:** the Streamlit file watcher walks transformers' lazy modules and logs `No module named 'torchvision'` tracebacks (harmless). Added `.streamlit/config.toml` with `fileWatcherType = "none"`, so **restart the app after code edits** (no hot reload).

### Dataset Step 1 — Photos (team, checked 2026-10-30)
- 94 photos, 11 categories (6–9 each), no exact duplicates. Target was 66 minimum, 130–160 recommended.
- Folders differ from the plan: `cleaning`, `furniture`, `network`, `plumbing`. They were left as they are; tools get the category and owner from the **file prefix**, not the folder.
- The team also filled in severity by hand in `review/images_<ID>.csv` before Step 2. Those are human labels and must not be overwritten. Their first versions are in `review/originals/`.
- Open issues: no `SOURCES.txt` anywhere, and many photos are web screenshots or stock images (Alamy, Adobe Stock, Shutterstock watermarks, Google Lens icons), which goes against DATASET_PLAN §4.1. Almost all are under the 640 px short-side guideline. F2 has 3 High/Critical photos (plan: ≥ 4).

### Dataset Step 2 — Image manifest + review sheets ✅ (review closed 2026-11-01)
- `tools/build_image_manifest.py`; run with `.\.venv\Scripts\python.exe -m tools.build_image_manifest`. Re-running is free for cached photos and is how edited sheets get merged into `manifests/images.csv`.
- Deviations from the plan:
  1. **Batches of 8 photos per call** (~13 calls instead of 92) because of the 20/day quota. Each photo is still sent once, categories are mixed across batches, and Gemini doesn't see the team's labels.
  2. Unlisted photos get `source`/`license` = `unknown`, not the plan's `own`, because most clearly aren't the team's own.
  3. Existing `severity`, `verified`, `source`, `license` and `category` are kept on re-run; short category names are mapped to `config.CATEGORIES` values.
  4. Added a `check_notes` column and `review/lead_flags.csv` (`image_id,hold,note`, kept by hand). `hold=yes` keeps a photo away from Gemini. Currently held for faces: `in_006`, `in_009`. Also noted: `sn_002`/`ch_003` are the same sink (a crop, which the hash check can't catch).
- Result: 92 of 94 analyzed (gemini-3.5-flash-lite, 8 on 3.8-flash). Team vs Gemini: severity agrees on 36% (87% within one level; 12 photos are 2 levels apart), category on 82% (many are Other/Transport vs Infrastructure). Sampled descriptions matched their photos, so answers aren't misaligned across a batch. Gemini's watermark flag caught only 11 photos, fewer than are visibly watermarked, so don't rely on it.
- Verified: 0 team severities changed; re-run made 0 Gemini calls and kept edits (lowercase severity, `Yes`, a source, a cp1252 save).
- Review outcome (2026-11-01): the lead confirmed the review and asked Claude to fill `verified=yes` on all 94 rows. Only 1 severity changed from the team's first labels (`ot_001` Critical → Medium). `in_006` and `in_009` were cropped (checked by eye, no faces left) and analyzed. Source and licence stay `unknown` for all 94 by the lead's decision, so this must be disclosed in any report.
- `sn_002`/`ch_003` were kept by the lead's decision, but they are still the same sink, so `lead_flags.csv` has `force_train=yes` for both.

### Dataset Step 3 — Complaint texts ✅
- `tools/generate_complaints.py`; run with `.\.venv\Scripts\python.exe -m tools.generate_complaints`. It refuses to run while any photo is unverified, on hold, unanalyzed or badly labelled (tested on the real unverified sheets). Caches are in `manifests/cache/complaints/`. Run it from the project root.
- Output: `manifests/complaints_draft.csv`, **436 rows**: 376 from photos (94 × 4 variants, 94 contradicting) + 60 text-only (Network 20, Transport 20, Security 10, Other 10). Train 340 / test 96 rows (73 / 21 photos, about 22%; text-only split by row). Every plan minimum in §2 is met. Labels match the sheets exactly, no photo is in both splits, 0 duplicate texts, 0 Hindi words.
- `manifests/splits.json` freezes the split on first run. Delete it only if reviewers haven't started on test rows.
- Deviations from the plan:
  1. **Batches of 4 photos per call** instead of 1 (quota). A batch that returns malformed JSON is retried one photo at a time, because some batches failed every time while the same photos worked alone. That retry path hasn't been exercised: the last failing batch passed on another model.
  2. Added a `contradiction_reason` column and a `gemini_model` column.
  3. **Contradicting rows go through a Gemini complaint check** (`audit_contradicting`, cached as `complaint_checked`), which rewrites any text saying nothing is wrong. Reason: those rows keep the photo's labels, and the first draft had many like "all the handrails are firmly bolted". The check rejected 22 on its first pass. `gemini-3.5-flash-lite` keeps writing them, so finishing the last few needed `gemini-3.5-flash`.
- Known limits for Step 4 reviewers: `tr_005` ("freshly paved") passed the check but isn't a complaint; `sn_007` reads oddly; some rows are exaggerations in another category's territory (`ch_004` water pipe under Cleaning); some may not really contradict their photo (e.g. `wm_001`).
- **Step 4 skipped (lead's decision, 2026-11-01):** no text review and no cross-check, so there is no Cohen's κ. Texts are accepted as generated. Never mark these rows `human_verified`: only the photo labels were checked by people. Step 7 must use a distinct `label_source` (e.g. `photo_label_inherited`), relax the plan's "every test row human_verified and cross-checked" check, and the report must disclose this. The 60 text-only rows (12 in test) have labels no person checked.
- **Phase 7 decision (2026-11-01):** contradicting rows keep the photo's severity by plan, so "only a minor pothole" can carry Critical. Default: leave `has_contradiction=True` rows out of **severity** training and keep them for **category** training; score both classifiers on non-contradicting test rows (75) and use the 21 contradicting test rows for contradiction evaluation. Confirmed by the lead the same day.
- Evidence (TF-IDF + logistic regression proxy on 75 clean test rows, not DistilBERT): category 54.7% with vs 52.0% without them (2 rows, within noise); severity 34.7% both ways. So exclusion rests on label correctness, not on a measured gain. Severity at 35% is low; re-check on real DistilBERT results.

### Dataset Step 5 — Voice recordings: scripts ready, team recording pending
- `tools/generate_recording_scripts.py` (no Gemini calls). Output: `data/real/scripts/<ID>.md`, `manifests/recording_lines.csv` (expected filename, reference text, env per speaker), `data/real/consent/README.md`. A re-run refuses once `recording_lines.csv` exists, because people record from it; delete it only if nobody has recorded.
- 24 lines per speaker, 63 distinct texts: 11 shared (one per category, all quiet, all test rows, 2 contradicting) + 13 own (from other members' categories, no overlap between speakers, 3 contradicting, 7 noisy). Every speaker covers all 11 categories and 4 severities.
- `EXCLUDED_ROWS` in the tool keeps out rows known to need rewriting (`tr_005`, `sn_007`, `wm_001`, `ch_004` contradicting). **Step 7 may rewrite those rows but must not change any text in `recording_lines.csv`.**
- **For Step 6:** informal lines contain "r", "n", "&", "pls" and typos, and speakers are told to say what was meant, so WER must normalize these in the reference or it will count correct speech as errors. Also accept `.mp3`/`.wav`, since the filenames show `.m4a`.
- `data/real/consent/` is git-ignored, so the blank template isn't in git either.
- **Recordings received 2026-11-03:** 94 clips in `data/real/audio/<ID>/` (M1 24, M2 24, F1 24, F2 22; F2 skipped `nw_004_detailed` and `wm_005_detailed`). All 4 speakers recorded all 11 shared lines. All clips are AAC, 3.7–15.2 s (median 7.8 s), with no `_free` clips.
- The team uploaded them to `data/real/<ID>/` with mangled extensions (`.m4a.m4a.mp4`, `.m4a(quiet).m4a.mp4`). Claude moved and renamed them to `<ID>_<record_id>_<env>.m4a` (same AAC audio; `.mp4` → `.m4a` is only a container rename) and deleted one byte-identical duplicate (`F2_wm_009 … (1).mp4`). The original → new mapping is in `manifests/audio_renames.csv`. Every filename matched its script line and environment.
- **Consent:** `data/real/consent/` still holds only the blank template, with no signed forms. DATASET_PLAN §8.4 requires 4 signed forms.

### Dataset Step 6 — Audio processing ✅
- `tools/process_audio.py`; run with `.\.venv\Scripts\python.exe -m tools.process_audio`. Runs locally with no Gemini calls. Transcripts are cached per setup + file hash in `manifests/cache/audio/`, so re-scoring takes seconds. Output `manifests/audio.csv` has the plan's columns plus `part`, `split`, `reference_words`, `*_errors`, and transcript/language/WER for each comparison setup.
- Compares 3 setups on the same 94 clips: `app` (= `speech_service.transcribe_audio`, Whisper base, auto language), `forced_en` (base, `language="en"`) and `base_en` (Whisper `base.en`, downloaded 139 MB to `~/.cache/whisper`).
- WER = word edit distance after normalizing both sides: Whisper's `EnglishTextNormalizer`, plus chat-abbreviation expansion in the reference (r/n/pls/&…), uniform times ("8:30"/"8.30", "a.m."), and a `SPELLINGS` map applied to both sides (asap, wifi, cant, split compounds). Real mishearings ("block c" → "block sea") still count. Speakers' own additions ("so please check that") count too, as the plan says to keep deviations.
- **Results (corpus WER):** app 12.5%, forced_en 12.5%, base_en 12.2%. Per speaker: M2 8.8%, M1 13.1%, F1 13.6%, F2 14.3%. Shared lines only (same 11 texts, all quiet): M1 17.2%, M2 8.3%, F1 13.2%, F2 15.7%. Informal lines 16.3% vs detailed 11.8%.
- **Interpretation:** the speaker differs more than gender does (the two male speakers are 8.3% and 17.2% on the same lines), so with 2 speakers per gender **don't claim a gender effect**. Noisy (10.2%) looks better than quiet (13.3%), but noisy clips are all "own" lines with different texts, so that comparison is confounded and shouldn't be reported as a noise effect.
- **Decision: keep the app as built (base, auto-detect).** Forcing English changed only the 3 clips where M1 was detected as Hindi, and they tied overall. `base.en` is 0.3 points better overall but worse for F2, which is within noise.

### Dataset Step 7 — Split, validate, export ✅
- `tools/finalize_dataset.py`; run with `.\.venv\Scripts\python.exe -m tools.finalize_dataset`. No Gemini calls. If validation fails, it exports nothing.
- Output:
  - `data/real/train.csv` (340 rows) and `data/real/test.csv` (96 rows): the Swap-Guide columns `raw_complaint, category, severity, has_contradiction`, plus `record_id, image_id, owner, variant, label_source, text_source`.
  - test.csv also has project-relative `image_file` (84 rows) and `audio_files` (`;`-separated; 25 test rows have 58 clips).
  - `manifests/complaints_reviewed.csv` (all rows with both media columns) and `data/real/DATASET_CARD.md`: counts, provenance, WER, licences, consent count. Use the card in the report.
- `label_source`: `photo_label_inherited` (376) / `seed_label_unchecked` (60). Never `human_verified`.
- `REWRITES` in the tool replaced 4 bad contradicting texts (`tr_005`, `sn_007`, `wm_001`, `ch_004`, all in train), written by Claude from the photos, with `text_source=claude_rewrite`.
- Validation is the plan's list. The plan's "test rows human_verified + cross-checked" check is replaced by "every photo row comes from a verified photo". Added checks: force_train photos not in test, every category and severity in test, recorded texts unchanged, no orphan audio. Each check was tested against deliberately broken data and caught it.
- Train severity counts without contradicting rows: Medium 116, Low 64, High 56, Critical 31.
- **Next = Phase 7 retrain on real data.** `models/classifier.py` needs a change for the agreed rule: drop `has_contradiction=True` rows for the severity model only. The current models in `models/saved/` are the synthetic ones; back them up before retraining if the demo needs them. Then extend Phase 8 to evaluate on `data/real/test.csv` (non-contradicting rows for classification; contradicting + image rows for contradiction detection; `audio_files` for the audio modality).
