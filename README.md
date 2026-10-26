# CivicFix

Multimodal GenAI civic complaint structuring system: text, voice and a photo go in; a structured, evidence-checked complaint comes out.

## Setup
pip install -r requirements.txt
cp .env.example .env  # add GEMINI_API_KEY

## Train custom models (optional — pre-trained fallback available)
python -m models.data_generator
python -m models.relabel_severity
python -m models.classifier --train

## Run
streamlit run app.py

## Evaluate
python -m eval.eval_runner

## Notes
- Custom models (DistilBERT, MiniLM) are trained on **synthetic** data; if `models/saved/` is missing the app runs Gemini-only.
- Free-tier Gemini allows 20 requests/day per model. The app falls back across `GEMINI_FALLBACK_MODELS` and, by default, reuses cached results for identical inputs (`demo_cache/`, toggle in the sidebar).
- Demo inputs and backup transcripts: `sample_data/` and `sample_data/voice_samples.md`.
