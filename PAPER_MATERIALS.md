# CivicFix: Material for the Research Paper and Report

Everything needed to write the CivicFix paper (IEEE conference format) and the project report, in one place.
Every number below comes from the project's result files. File paths are given so any number can be re-checked.

**How to use this file**
- §1–§3: title, abstract, contributions, problem. Ready to adapt.
- §4–§6: system, dataset and models. These become the paper's *System*, *Dataset* and *Method* sections.
- §7–§8: evaluation set-up and results, with ready-to-paste IEEE LaTeX tables (§14).
- §9–§11: discussion, limitations and ethics, future work. **Read §10 before writing any claim.**
- §12: commands and software versions, for reproducibility.
- §13: references to verify before citing.
- Labels such as M1, M2, F1, F2 are anonymous team-member IDs. Use them, or replace with "Annotator 1…4". Never real names.

**Status of the work (2026-11-03)**

| Part | Status |
|---|---|
| Working app (text + voice + photo → structured, evidence-checked complaint) | Done |
| Real dataset (94 photos, 436 texts, 94 voice clips) | Done |
| DistilBERT trained on real data | Done |
| Evaluation: category/severity, contradiction detection, speech recognition | Done |
| Evaluation: modality ablation (text vs photo vs voice vs all) | **Not yet run.** Leave out, or list as future work |
| Evaluation: hallucination with/without audit **on real data** | **Not yet run.** Only a 15-case pilot exists (§9.5) |

---

## 1. Title, abstract, keywords

**Title options**
1. *CivicFix: Evidence-Grounded Multimodal Structuring of Civic Complaints with Contradiction Auditing*
2. *Do Photos Keep Complaints Honest? Multimodal Claim Verification for Campus Maintenance Reports*
3. *From Informal Reports to Actionable Complaints: A Multimodal Pipeline with LLM and NLI-Based Contradiction Detection*

**Draft abstract (about 200 words; adjust once the final results are fixed)**

> Campus and civic maintenance complaints are typically short, informal and incomplete, and they are sometimes exaggerated, which makes them hard to route and prioritise. We present CivicFix, a multimodal pipeline that turns any combination of typed text, a voice note and a photo into a structured complaint with a category, severity, responsible department and formal complaint text. Before writing, the pipeline audits the user's claims against the visual evidence, and it asks the user to resolve mismatches. We built a real evaluation dataset: 94 photos of campus problems across 11 categories with team-verified labels, 436 complaint texts grounded in those photos (including 94 deliberately contradicting texts), and 94 voice recordings from four speakers. On contradiction detection, a Gemini-based audit reached F1 0.95 (precision 1.00, recall 0.90) with no false alarms on matching photo–text pairs. A pre-trained MiniLM NLI cross-encoder reached perfect recall but raised false alarms on 29% of honest complaints. A fine-tuned DistilBERT matched the zero-shot LLM on text-only category classification (61.3% vs 58.7%; overlapping confidence intervals). We find that severity is not recoverable from text alone (≤39% accuracy for every system), because severity labels derive from the photo. Local Whisper transcription reached a word error rate of 12.5%.

**Keywords:** multimodal NLP, civic technology, complaint classification, contradiction detection, natural language inference, hallucination, large language models, speech recognition.

---

## 2. Contributions (for the end of the Introduction)

1. **A multimodal complaint-structuring pipeline** combining local speech recognition (Whisper), an LLM (Gemini) for structured extraction, a contradiction audit, and grounded complaint generation. It includes a human-in-the-loop step when the photo and description disagree.
2. **A real, multimodal evaluation dataset**: 94 photos across 11 categories with team-verified severity, 436 photo-grounded texts in 4 controlled variants (including deliberate contradictions), and 94 recordings from 4 speakers (2 male, 2 female). It has a photo-grouped train/test split to prevent leakage.
3. **A paired contradiction benchmark**: each test photo is evaluated with a contradicting text and with a matching text. This measures both detection and false alarms. It compares an LLM audit with a pre-trained NLI cross-encoder.
4. **An honest comparison of a fine-tuned small model against a zero-shot LLM**, with confidence intervals, and an analysis showing that the gap is largely one of taxonomy conventions, not language understanding.
5. **A finding about label provenance**: severity labels assigned from photos are not learnable from text. This has implications for text-only complaint triage systems.

---

## 3. Problem statement and objectives

**Problem.** Complaints such as a dead streetlight, an overflowing bin or a leaking pipe usually arrive as short, informal messages ("light not working since 3 days"). They often lack a location, duration or severity, and are sometimes exaggerated. Maintenance staff cannot route or prioritise them reliably. A photo can supply evidence, but it can also contradict what the person wrote.

**Research objectives** (from the project plan)
1. A multimodal structuring pipeline using both pre-trained and custom-trained models.
2. Contradiction detection: cross-check user claims against image evidence.
3. Hallucination reduction: measure the unsupported-sentence rate with and without the audit stage.
4. Measurable evaluation: classification accuracy, contradiction precision/recall, a deterministic quality score, and a modality ablation.

**Research questions** (suggested framing for the paper)
- **RQ1:** Can a contradiction audit detect complaints that their own photo disproves, without flagging honest ones?
- **RQ2:** Does a small model fine-tuned on in-domain data match a zero-shot LLM at categorising complaints?
- **RQ3:** Can severity be predicted from complaint text alone?
- **RQ4:** How accurate is local speech recognition on real, informal spoken complaints?

---

## 4. System architecture

### 4.1 Pipeline (Figure 1, to be drawn as a block diagram)

```
Voice note ──► Whisper (local, base) ──► transcript ─┐
Typed text ──────────────────────────────────────────┼─► Stage 1: Gemini structured analysis ──► StructuredIssue (14 fields)
Photo ───────────────────────────────────────────────┘        (text + transcript + image)              │
                                                                                                         ▼
                                           Stage 2: contradiction audit: claims vs photo evidence
                                           (Gemini audit; MiniLM NLI shown side by side)
                                                   │                                   │
                                     photo matches │                                   │ photo contradicts / mismatches
                                                   ▼                                   ▼
                              Stage 3: Gemini complaint from          User review: re-upload photo / edit text /
                              VERIFIED facts only                     continue → re-analyse WITHOUT the photo and write a
                                                   │                  flagged complaint from the user's own facts
                                                   ▼
                              Deterministic quality score (no LLM) + downloadable report
            In parallel: DistilBERT predicts category + severity from the text (shown next to Gemini's answer)
```

### 4.2 Components

| # | Function | Implementation |
|---|---|---|
| 1 | Speech-to-text (local; audio never leaves the machine) | OpenAI Whisper `base`, language auto-detected, CPU |
| 2 | Structured extraction (text + transcript + photo → 14-field JSON) | Google Gemini, temperature 0.1, output validated by a Pydantic schema |
| 3 | Evidence separation | Four separate lists: `observed_evidence` (what the photo shows), `user_reported_facts`, `inferred_information`, `missing_information` |
| 4 | Contradiction audit (LLM) | Gemini, temperature 0.0. Labels each claim *supported*, *contradicted* or *unverifiable*, and states whether the photo matches the report at all |
| 5 | Contradiction audit (non-LLM) | Pre-trained cross-encoder `cross-encoder/nli-MiniLM2-L6-H768`: premise = photo evidence, hypothesis = claim; entailment → supported, neutral → unverifiable, contradiction → contradicted |
| 6 | Category classifier (custom) | Fine-tuned `distilbert-base-uncased`, 11 classes |
| 7 | Severity classifier (custom) | Separately fine-tuned `distilbert-base-uncased`, 4 classes (Low / Medium / High / Critical) |
| 8 | Grounded complaint generation | Gemini, temperature 0.3, given only the audit's verified facts |
| 9 | Human-in-the-loop mismatch resolution | If the photo contradicts the text, the pipeline stops before writing. "Continue" re-analyses without the photo and writes a complaint marked as unconfirmed by the photo |
| 10 | Quality score | Rule-based (§7.4) |

**Why three stages instead of one prompt:** the audit needs claims and evidence separated first. Separate stages can each be validated, compared (Gemini vs MiniLM) and ablated (with vs without the audit). The pipeline can also stop for the user's decision before generating a complaint.

**Engineering details worth one sentence each:**
- Every LLM output is schema-validated and retried once on malformed JSON.
- Model fallback across several Gemini models handles quota and overload.
- Results are cached, so evaluation is reproducible without re-querying.

**Categories (11):** Electrical, Plumbing / Water, Sanitation, Waste Management, Infrastructure, Security, Cleaning / Housekeeping, Furniture / Facilities, Internet / Network, Transport, Other.

**Severity definitions** (used identically in prompts and for human labelling):
- **Low:** minor inconvenience, no safety impact, few users affected.
- **Medium:** meaningful inconvenience, maintenance needed, not urgent.
- **High:** significant impact on safety, accessibility, sanitation or essential services.
- **Critical:** immediate danger or a serious hazard requiring urgent response.

---

## 5. Dataset

Source of all numbers: `data/real/DATASET_CARD.md`, `data/real/manifests/*.csv`.

### 5.1 Collection procedure (7 steps)

1. **Photos.** Four team members each collected photos for the categories they owned: 94 photos in total, 6–9 per category.
   - **Privacy:** two photos were cropped to remove identifiable faces before any processing.
   - **Duplicates:** one near-duplicate pair (the same sink, cropped differently) was kept, and both copies were forced into the training split to prevent leakage.
2. **Photo labels.** Team members assigned the severity of each photo, and the category came from its folder.
   - Gemini then gave an independent second opinion, without seeing the team's labels: a scene description, a proposed severity and a category check.
   - Owners reviewed their sheets and marked every photo verified.
   - **Team–Gemini agreement:** 36% on exact severity (87% within one level) and 82% on category.
3. **Complaint texts.** For each photo, Gemini wrote 4 English variants as a student or resident would type them:

   | Variant | Style | Contradiction |
   |---|---|---|
   | consistent_informal | casual, abbreviations, typos | no |
   | consistent_detailed | well written, plausible location/duration | no |
   | vague | minimal information | no |
   | contradicting | a sincere complaint the photo **visibly disproves** (wrong fault, wrong scale, exaggeration) | **yes** |

   - Every text inherits its photo's verified category and severity, so Gemini never chose a label.
   - **Contradiction quality control:** many first-draft contradicting texts were not complaints at all ("all the handrails are firmly bolted"). A second Gemini pass rejected any contradicting text that didn't report a problem, and those texts were rewritten (22 on the first pass). Four that still slipped through were rewritten by hand.
   - **Text-only rows:** 60 extra rows cover problems that are hard to photograph: Internet/Network 20, Transport 20, Security 10, Other 10. Each was generated from a category + severity seed.
4. **Text review** was skipped. Texts were accepted as generated (see §10).
5. **Voice recordings.** Each of 4 speakers (2 male, 2 female) read a script of 24 lines.
   - **Shared lines:** 11 lines were the same for everyone (one per category, all recorded in a quiet room), to compare speakers on identical text.
   - **Own lines:** 13 more per speaker, taken from other members' categories with no overlap between speakers; 7 of them recorded in a noisy setting.
   - **Result:** 94 clips, 12.7 minutes in total, 3.7–15.1 s each (median 7.8 s). One speaker recorded 22 of their 24 lines.
6. **Transcription and scoring:** Whisper transcription and word error rate (§7, §8.3).
7. **Split and validation.**
   - **Split:** about 20% of photos went to test, chosen per category to cover all severities, and **all texts of a photo stay in the same split**. Text-only rows are split by row.
   - **Automatic checks:** files exist, no duplicate texts, labels valid, no photo in both splits, minimum class counts met, recorded texts unchanged.

### 5.2 Statistics (Table I)

| | Train | Test | Total |
|---|---|---|---|
| Complaint texts | 340 | 96 | 436 |
| … from photos (4 variants × photo) | 292 (73 photos) | 84 (21 photos) | 376 (94 photos) |
| … text-only | 48 | 12 | 60 |
| Contradicting texts | 73 | 21 | 94 |
| Voice clips (of rows in this split) | 36 | 58 | 94 |

**Per severity (texts):** Low 102, Medium 189, High 94, Critical 51.
**Per photo (severity):** Low 22, Medium 42, High 19, Critical 11.
**Per category (texts):** Transport 52, Security 46, Other 46, Internet/Network 44, Cleaning 36, Electrical 36, Infrastructure 36, Plumbing/Water 36, Sanitation 36, Waste Management 36, Furniture/Facilities 32. The largest category is 12% of the data.
**Per variant:** consistent_informal 112, consistent_detailed 114, vague 116, contradicting 94.
**Text length:** 3–33 words (median 13).

**Example texts from one photo (`el_004`, a damaged cable with exposed copper; Electrical, Critical):**
- informal: *"urgent!! live wire totally chewed up and copper strands coming out near lab corridor, pls cut power and fix asap"*
- vague: *"Dangerous wire issue here, needs attention immediately before accident happens."*

### 5.3 Label provenance (state this in the paper)

| label_source | Rows | Meaning |
|---|---|---|
| photo_label_inherited | 376 | Text written by Gemini; category and severity from the team-verified photo |
| seed_label_unchecked | 60 | Text-only row; labels are the generation seed, and no person checked them |

Contradicting rows keep the photo's severity by design. Their severity label therefore does not describe their text, so they are **excluded from severity training and severity scoring**.

---

## 6. Models and training

### 6.1 DistilBERT classifiers (category and severity)

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` (Hugging Face), one model per task |
| Max sequence length | 128 tokens |
| Optimiser / learning rate | AdamW, 5e-5, linear decay, no warm-up, no weight decay (Hugging Face `Trainer` defaults) |
| Batch size / epochs | 16 / 15, best epoch kept |
| Validation | 20% of `train.csv`, **split by photo** (StratifiedGroupKFold), so the texts of one photo are never in both halves |
| Best-epoch criterion | category: accuracy; severity: macro-F1 |
| Severity training data | contradicting rows removed (267 rows: Medium 116, Low 64, High 56, Critical 31) |
| Severity loss | class-weighted cross-entropy (balanced weights) |
| Hardware | CPU only, about 5 min per model |
| Seed | 42 |

**Why class weights for severity:** an unweighted model collapsed to predicting "Medium" for every input. That gave 41.3% test accuracy (the majority-class rate) but zero recall on Low, High and Critical. The weighted model has slightly lower accuracy and a much higher macro-F1 (0.146 → 0.357). It is the one reported.

**Why photo-grouped validation:** with several texts per photo, a random split puts near-paraphrases of the same scene in both training and validation. That inflates validation scores and biases the choice of the best epoch.

### 6.2 MiniLM NLI cross-encoder
- `cross-encoder/nli-MiniLM2-L6-H768`, used **pre-trained, not fine-tuned**.
- Input: (premise = the photo evidence sentences joined, hypothesis = one user claim). The argmax label gives contradiction / entailment / neutral.
- A complaint counts as flagged if any claim is labelled *contradiction*.
- In the evaluation it sees exactly the same extracted claims and evidence as the Gemini audit.

### 6.3 Gemini
- Used for analysis (temperature 0.1), auditing (0.0) and complaint writing (0.3), plus dataset generation.
- **Models served in the evaluation:** `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`. Free-tier limits caused automatic fallback between them, and the model that answered is recorded for every case in `eval/results/real_eval_cases.csv`.
- State this in the paper: LLM versions change, so results may not reproduce exactly with other models.

### 6.4 Whisper
`base` multilingual model, local CPU inference, `fp16=False`. This is the app's configuration. Two alternatives were compared (§8.3).

### 6.5 Deterministic quality score (no LLM)
- **Overall** = mean of 4 dimensions (each 0–100):
  - **Completeness:** problem 25, category 20, location 25, duration 15, department 15.
  - **Specificity:** problem ≥4 words 30, subcategory 20, severity reason ≥5 words 25, action ≥5 words 25.
  - **Evidence:** 20 per photo-evidence item (max 50), 10 per user fact (max 30), +20 if there are no contradictions, otherwise −10 per contradiction.
  - **Actionability:** department 30, recommended action 35, High/Critical 20, nothing missing 15.
- **Raw-input score:** text length up to 30, location word 15, duration word 10, photo 25, voice 20.
- It is shown as a before (raw input) vs after (generated report) comparison.

---

## 7. Evaluation set-up

All evaluation uses the held-out `data/real/test.csv`; no test photo was seen in training. Code: `eval/real_eval.py`.

| Experiment | Cases | Protocol | Metrics |
|---|---|---|---|
| **E1: category and severity** | 75 non-contradicting test texts | Text only, the same input to both systems. Gemini runs the app's analysis stage; its "Main / Sub" category strings are normalised to the 11 classes | Accuracy with a 95% Wilson confidence interval, macro-F1, weighted F1, per-class recall |
| **E2: contradiction detection** | 42 = 21 test photos × {contradicting text, matching informal text} | Each text is analysed **with its photo**. Gemini is flagged if the audit marks any claim contradicted **or** says the photo doesn't match: exactly the app's rule for stopping and asking the user. MiniLM is flagged if any claim's NLI label is contradiction | Precision, recall, F1 and accuracy for "contradicted", plus the false-alarm rate on the 21 matching texts |
| **E3: speech recognition** | 94 clips | Word error rate against the script line | Corpus-level WER (total errors / total reference words) |

**WER normalisation** (both sides):
- Whisper's English text normaliser handles numbers, punctuation and spelling variants.
- Chat abbreviations in the scripts were expanded ("r"→"are", "pls"→"please", "&"→"and"), because speakers were told to say what was meant.
- Times were made uniform ("8:30" = "8.30"; "a.m." = "am").
- Spelling-only variants were unified ("asap" = "as soon as possible", "wifi" = "Wi-Fi", split compounds).
- Real mishearings still count ("block c" → "block sea"), and so do words speakers added.

**Why paired controls in E2:** they measure false alarms, not just detection. A detector that flags everything gets recall 1.0 on positives alone.

---

## 8. Results

### 8.1 E1: text-only category and severity (Table II)

| Task | System | Accuracy [95% CI] | Macro-F1 | Weighted F1 |
|---|---|---|---|---|
| Category | Gemini (zero-shot) | 58.7% [47.4, 69.1] | 0.623 | 0.614 |
| Category | DistilBERT (fine-tuned) | **61.3%** [50.0, 71.5] | 0.587 | 0.599 |
| Severity | Gemini (zero-shot) | 36.0% [26.1, 47.3] | 0.349 | 0.345 |
| Severity | DistilBERT (fine-tuned, class-weighted) | **38.7%** [28.5, 50.0] | **0.357** | 0.370 |
| Severity | Majority class ("Medium") | 41.3% | 0.146 | n/a |

n = 75. The confidence intervals overlap heavily: **neither system is significantly better on either task.**

**More detail:**
- **Who is right where (category):** both wrong on 13 cases, only Gemini right on 16, only DistilBERT right on 18.
- **Category recall by class (Gemini / DistilBERT):**
  - Infrastructure 1.00 / 0.00, Waste Management 1.00 / 1.00, Plumbing 1.00 / 0.67, Sanitation 0.83 / 0.67;
  - Other 0.12 / 0.75, Internet/Network 0.57 / 1.00, Electrical 0.50 / 0.83, Security 0.38 / 0.62;
  - Furniture 0.17 / 0.17, Transport 0.50 / 0.60, Cleaning 0.67 / 0.33.
- **28 of Gemini's 31 category errors** were *Infrastructure* for texts the team filed under Transport (roads), Security (walls, fences), Other (fallen trees, signposts), Furniture, Electrical or Network. Example: *"projector hanging loosely from ceiling, might fall down anytime"* is labelled Furniture / Facilities, and Gemini said Infrastructure.
- **Category accuracy by text variant (Gemini / DistilBERT):** informal 56.5% / 60.9%, detailed 51.9% / 63.0%, vague 68.0% / 60.0%.
- **Severity within one level of the label:** Gemini 84.0%, DistilBERT 81.3%.
- **Severity recall (Low / Medium / High / Critical):** Gemini 0.19 / 0.29 / 0.63 / 0.33; DistilBERT 0.81 / 0.29 / 0.26 / 0.22.
- **Gemini rates severity higher than the team:** 17 of 31 Medium texts were called High.
- **Earlier reference points on the same test rows:**
  - DistilBERT trained on the earlier synthetic data scored 49.3% category and 24.0% severity, which shows the value of real training data.
  - A TF-IDF + logistic-regression baseline scored 56.0% category and 34.7% severity (macro-F1 0.278).

### 8.2 E2: contradiction detection with photos (Table III)

| System | Precision | Recall | F1 | Accuracy | False alarms on matching texts |
|---|---|---|---|---|---|
| Gemini audit (app rule) | **1.00** | 0.90 | **0.95** | **95.2%** | **0%** (0/21) |
| MiniLM NLI (pre-trained) | 0.78 | **1.00** | 0.88 | 85.7% | 29% (6/21) |

21 contradicting + 21 matching texts on the same 21 test photos. Gemini: TP 19, FP 0, FN 2. MiniLM: TP 21, FP 6, FN 0. Two of Gemini's catches came only through the "photo doesn't match the report" signal.

**Examples caught by Gemini (for a figure or table of qualitative results):**

| Photo (what it shows) | Contradicting text | Gemini's reason |
|---|---|---|
| A working, lit fluorescent fixture with peeling paint nearby (`el_001`) | "Both tube light bulbs are completely shattered … hanging by a single loose wire in total darkness." | "…whereas the image shows an intact, illuminated fluorescent light fixture." |
| A detached, rusted handrail on concrete stairs (`ff_006`) | "the entire concrete staircase has completely collapsed into a pile of rubble…" | "…the image shows the staircase structure remains intact with only a detached handrail." |
| A bus shelter with shattered glass (`tr_002`) | "the bus shelter structure is completely submerged in deep floodwater…" | "…the image shows structural damage and shattered glass with no evidence of water." |

- **Missed by Gemini:**
  - `nw_004`: a collapsed server rack, with the text "router antenna is snapped off … dead power adapter". The claim is a different fault in the same scene of damaged network equipment.
  - `wm_009`: scattered plastic litter, with the text "someone dumped massive iron furniture and construction debris". This is plausible clutter that the model didn't treat as disproven.
- **MiniLM's false alarms** were on uncertain or inferred claims rather than real conflicts. For example, it flagged "The projector may be held only by cables (uncertain)" and "Area lacks sufficient waste bins" on honest complaints.

### 8.3 E3: speech recognition (Table IV)

Corpus word error rate (lower is better):

| Group | Clips | Whisper base, auto language (app) | base, forced English | base.en |
|---|---|---|---|---|
| All clips | 94 | 12.5% | 12.5% | 12.2% |
| Speaker M1 | 24 | 13.1% | 13.1% | 12.6% |
| Speaker M2 | 24 | 8.8% | 8.8% | 8.8% |
| Speaker F1 | 24 | 13.6% | 13.6% | 11.1% |
| Speaker F2 | 22 | 14.3% | 14.3% | 16.5% |
| Shared lines only (same 11 texts per speaker) | 44 | 13.6% | 13.6% | 13.8% |
| Informal-style lines | 40 | 16.3% | 16.3% | 17.0% |
| Detailed-style lines | 34 | 11.8% | 11.8% | 12.6% |

- **Language detection:** Whisper detected Hindi on 3 clips of one speaker but still produced English text. Forcing English changed only those 3 clips and tied overall.
- **Typical errors** are short letters and words in context ("block c" → "block sea", "block b" → "bees") and words speakers added.
- **Shared lines per speaker:** M1 17.2%, M2 8.3%, F1 13.2%, F2 15.7%. Speakers of the same gender differ more than the genders do, so **no gender effect can be claimed** with 2 speakers per gender.
- **Do not report quiet vs noisy.** Noisy clips scored lower (10.2% vs 13.3%), but they used different, easier texts, so the comparison is confounded.

### 8.4 Pilot results (synthetic data; report only as a pilot)

An earlier 15-case, hand-written, text-only evaluation (`eval/results/eval_report.json`) used models trained on synthetic data:
- **Category:** Gemini 100%, DistilBERT 53%.
- **Severity:** Gemini 73%, DistilBERT 67%.
- **Contradiction:** without photos, Gemini flagged 0 of 3 exaggerated claims, because it marks text-only claims "unverifiable". This motivated the photo-based benchmark.
- **Hallucination proxy:** unsupported-sentence rate 43.3% with the audit vs 45.6% without. Two complaints were written from the same analysis, with and without the audit's verified facts, and both scored against the same grounded reference.
- **Quality:** average score 73.7.

These numbers are from 15 cases and synthetic training data. **Don't compare them directly with §8.1–8.3.**

---

## 9. Discussion (key points to develop)

1. **Photos make contradiction detection possible.** Without a photo, an LLM can only call exaggerated claims "unverifiable" (pilot: 0/3 caught). With the photo, the audit caught 19/21 with no false alarms. The LLM audit is precise; the off-the-shelf NLI model is sensitive but over-flags hedged or inferred claims. Fine-tuning MiniLM on the dataset's contradiction pairs is a natural next step.
2. **A fine-tuned small model matches a zero-shot LLM on category** (61.3% vs 58.7%, not significant), but for a different reason than raw capability. Gemini's errors are dominated by one systematic disagreement: it files roads, walls, fences and fallen trees under *Infrastructure*, while the team's taxonomy puts them under Transport, Security or Other. DistilBERT learned the team's conventions from its training data, and Gemini had only the category names. **Implication:** category boundaries need written guidelines, or examples in the prompt, before LLM accuracy can be judged.
3. **Severity is not recoverable from text alone here.** Every system (DistilBERT, Gemini, TF-IDF) scored at or below the majority-class rate of 41.3% on accuracy, and no subset of the data (detailed texts only, no vague texts, photo rows only) let even the baseline exceed about 44% / macro-F1 0.34. The labels were assigned by looking at the photo, while the classifiers read only the text. A text like "wall is dirty" carries whatever severity its photo received. Severity scales also differ between annotators and the LLM (36% exact agreement). **Implication:** severity triage should use the photo, or labels must be assigned from the text a system actually sees.
4. **Local speech recognition is usable but not perfect.** 12.5% WER on informal spoken complaints with the small `base` model. Most errors hit single letters and short words (block names, room letters), which matter for location. Larger Whisper models are an obvious improvement, at the cost of CPU time.
5. **Human-in-the-loop matters for honest reporting.** When the photo contradicts the description, the system asks the user instead of silently trusting one source. If the user continues, the complaint is clearly marked as unconfirmed by the photo.

---

## 10. Limitations and threats to validity (keep these in the paper)

- **Small test set.** 75 classification cases and 42 contradiction cases. The confidence intervals are wide, and per-class results rest on very few examples.
- **Texts were not individually reviewed by humans.** Complaint texts were written by Gemini, and their labels are inherited from human-verified photos. **Do not describe the test set as "human-verified".** There is no inter-annotator agreement score.
- **The 60 text-only rows** have seed labels no person checked.
- **Generated texts may favour Gemini**, which wrote them (stylistic familiarity). This could inflate Gemini's scores on E1/E2.
- **Contradictions are synthetic.** They are deliberately written by an LLM, which may be easier to catch than real users' exaggerations.
- **Image licensing.** Photo sources and licences are unknown, and many are web or stock images, some watermarked. **Don't publish the images**; describe them instead.
- **Few speakers.** 4 speakers, 2 per gender, reading scripts, plus a confounded noise condition.
- **LLM versions.** Results depend on the specific Gemini versions served (flash-lite models, free tier, with automatic fallback), which change over time.
- **Single site, English only.** The data comes from one campus context.
- **Not yet evaluated:** the modality ablation and the with/without-audit hallucination measure on real data.

**Ethics and privacy:**
- **Audio** is transcribed locally; only text transcripts are sent to the LLM.
- **Faces** were cropped out of photos before processing.
- **Identities:** speakers are identified by anonymous IDs only.
- **Consent:** written consent forms exist, but signed copies must be on file before the voice data is used in a publication.
- **Photos to Gemini:** photos are sent to Google's Gemini API, and free-tier data may be used by the provider.

---

## 11. Future work
- Run the modality ablation (text vs photo vs voice transcript vs all) on the test photos with their recordings, and the hallucination measure on real data.
- Fine-tune the NLI cross-encoder on the 94 contradiction pairs and compare it with the pre-trained model.
- Write category guidelines (e.g. "a road surface problem goes to Infrastructure, not Transport") and give the LLM few-shot examples. Re-measure the taxonomy gap.
- Predict severity from the photo (or photo + text) instead of text alone.
- Collect real user complaints instead of generated ones, review the text labels, and measure inter-annotator agreement.
- Try larger Whisper models, or ones fine-tuned for English accents.

---

## 12. Reproducibility

| What | Command (from the project folder, Windows PowerShell) |
|---|---|
| Run the app | `.\.venv\Scripts\python.exe -m streamlit run app.py` |
| Build image manifest / review sheets | `.\.venv\Scripts\python.exe -m tools.build_image_manifest` |
| Generate complaint texts | `.\.venv\Scripts\python.exe -m tools.generate_complaints` |
| Generate recording scripts | `.\.venv\Scripts\python.exe -m tools.generate_recording_scripts` |
| Transcribe + WER | `.\.venv\Scripts\python.exe -m tools.process_audio` |
| Validate + export dataset | `.\.venv\Scripts\python.exe -m tools.finalize_dataset` |
| Train category model | `.\.venv\Scripts\python.exe -m models.classifier --train --data data/real/train.csv --only category` |
| Train severity model | `.\.venv\Scripts\python.exe -m models.classifier --train --data data/real/train.csv --only severity --severity-skip-contradicting --balanced-severity` |
| Evaluate on the real test set | `.\.venv\Scripts\python.exe -m eval.real_eval` |

**Software:** Python 3.13.7; transformers 5.20.2; torch 2.14.1 (CPU); sentence-transformers 6.2.0; openai-whisper 20250625; google-genai 2.33.0; scikit-learn 1.9.1; pandas 3.1.0; Streamlit 1.66.1.

**Result files:**
- Dataset: `data/real/train.csv`, `data/real/test.csv`, `data/real/DATASET_CARD.md`.
- Evaluation: `eval/results/real_eval_report.json` (all metrics) and `eval/results/real_eval_cases.csv` (every case with each system's answer).
- Speech recognition: `data/real/manifests/audio.csv` (every clip with its transcripts).

---

## 13. References to cite (verify each entry before using it)

These are standard references for the components used. Check every detail (authors, venue, year) in Google Scholar, and format them in IEEE style. Add 4–8 related-work papers found through your own search: civic issue reporting (311 systems, FixMyStreet), complaint classification, multimodal misinformation / claim verification, and LLM hallucination.

1. A. Radford et al., "Robust speech recognition via large-scale weak supervision," *ICML*, 2023. (Whisper)
2. V. Sanh, L. Debut, J. Chaumond, T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.
3. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," *NAACL-HLT*, 2019.
4. W. Wang et al., "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," *NeurIPS*, 2020.
5. N. Reimers, I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," *EMNLP-IJCNLP*, 2019. (sentence-transformers library, cross-encoders)
6. S. R. Bowman et al., "A large annotated corpus for learning natural language inference," *EMNLP*, 2015. (SNLI)
7. A. Williams, N. Nangia, S. R. Bowman, "A broad-coverage challenge corpus for sentence understanding through inference," *NAACL-HLT*, 2018. (MultiNLI)
8. Gemini Team, Google, "Gemini: A family of highly capable multimodal models," arXiv:2312.11805, 2023.
9. Z. Ji et al., "Survey of hallucination in natural language generation," *ACM Computing Surveys*, 2023.
10. J. Thorne et al., "FEVER: a large-scale dataset for fact extraction and VERification," *NAACL-HLT*, 2018. (claim verification)
11. E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, 1927. (confidence interval used)
12. T. Wolf et al., "Transformers: State-of-the-art natural language processing," *EMNLP: System Demonstrations*, 2020. (Hugging Face)
13. Model card: `cross-encoder/nli-MiniLM2-L6-H768`, Hugging Face Hub (cite as an online resource with access date).

---

## 14. Ready-to-paste IEEE LaTeX tables

These need `\usepackage{booktabs}` in the preamble.

```latex
\begin{table}[t]
\caption{Dataset statistics}
\label{tab:dataset}
\centering
\begin{tabular}{lrrr}
\toprule
 & Train & Test & Total \\
\midrule
Complaint texts & 340 & 96 & 436 \\
\quad from photos (4 variants each) & 292 & 84 & 376 \\
\quad text-only & 48 & 12 & 60 \\
Photos & 73 & 21 & 94 \\
Contradicting texts & 73 & 21 & 94 \\
Voice clips & 36 & 58 & 94 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\caption{Text-only classification on the real test set ($n=75$). Accuracy with 95\% Wilson interval.}
\label{tab:classification}
\centering
\begin{tabular}{llcc}
\toprule
Task & System & Accuracy [95\% CI] & Macro-F1 \\
\midrule
Category & Gemini (zero-shot) & 58.7 [47.4, 69.1] & 0.623 \\
Category & DistilBERT (fine-tuned) & \textbf{61.3} [50.0, 71.5] & 0.587 \\
\midrule
Severity & Gemini (zero-shot) & 36.0 [26.1, 47.3] & 0.349 \\
Severity & DistilBERT (class-weighted) & \textbf{38.7} [28.5, 50.0] & \textbf{0.357} \\
Severity & Majority class & 41.3 & 0.146 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\caption{Contradiction detection with photos: 21 contradicting and 21 matching texts on the same 21 test photos}
\label{tab:contradiction}
\centering
\begin{tabular}{lccccc}
\toprule
System & P & R & F1 & Acc. & False alarms \\
\midrule
Gemini audit & \textbf{1.00} & 0.90 & \textbf{0.95} & \textbf{95.2} & \textbf{0/21} \\
MiniLM NLI & 0.78 & \textbf{1.00} & 0.88 & 85.7 & 6/21 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\caption{Word error rate (\%) of Whisper on 94 spoken complaints}
\label{tab:wer}
\centering
\begin{tabular}{lrccc}
\toprule
Group & Clips & base (auto) & base (en) & base.en \\
\midrule
All & 94 & 12.5 & 12.5 & 12.2 \\
Speaker M1 & 24 & 13.1 & 13.1 & 12.6 \\
Speaker M2 & 24 & 8.8 & 8.8 & 8.8 \\
Speaker F1 & 24 & 13.6 & 13.6 & 11.1 \\
Speaker F2 & 22 & 14.3 & 14.3 & 16.5 \\
Shared lines & 44 & 13.6 & 13.6 & 13.8 \\
\bottomrule
\end{tabular}
\end{table}
```

**Suggested figures**
- **Fig. 1:** pipeline block diagram (§4.1).
- **Fig. 2:** app screenshot of the mismatch review (the photo contradicts the description, with the three choices).
- **Fig. 3:** category confusion matrix for Gemini vs DistilBERT, highlighting the "Infrastructure" column. It can be built from `eval/results/real_eval_cases.csv`.
- **Fig. 4:** qualitative contradiction examples (§8.2 table), described in words rather than reproducing the photos, because of licensing.

**Suggested paper outline (IEEE conference, 6 pages)**
1. Introduction: §3, contributions §2.
2. Related work: own search plus §13.
3. System: §4.
4. Dataset: §5.
5. Models and experimental set-up: §6–§7.
6. Results: §8.1–8.3, Tables II–IV.
7. Discussion: §9.
8. Limitations and ethics: §10.
9. Conclusion and future work: §11.
