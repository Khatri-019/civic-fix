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
| 6R — Real dataset (DATASET_PLAN.md) | Deferred until after the demo | |
| 7 — DistilBERT + MiniLM | ✅ Done (retrained: cat 80%, sev 73% val) | 2026-10-26 |
| 8 — Evaluation Suite | ✅ Done | 2026-10-26 |
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
