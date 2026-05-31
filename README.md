# Payer Policy Intelligence Pipeline

Extracts 13 Prior Authorization (PA) policy parameters from payer policy PDFs for Plaque Psoriasis (PsO) indications and computes a 0–100 Access Score per row.

> **Dashboard:** [https://pa-policy-predictor.streamlit.app](https://pa-policy-predictor.streamlit.app/)
>
> The dashboard is not just a viewer — it is a full interactive UI where you can run extractions, inspect PDFs, explore results, and download outputs. For custom data or local use, see [Running Locally](#running-locally) below.

<!-- Replace the URL above once deployed -->

---

## Submission Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | **results.xlsx** — one row per Filename+Brand, all 13 parameters + Access Score | `pipeline/output/results.xlsx` (also `results.csv`) |
| 2 | **Codebase** — full pipeline from PDF ingestion to structured output | This `pipeline/` directory |
| 3 | **Access Score write-up** | [Below](#access-score-framework) |
| 4 | **Dashboard + Screenshots** | Streamlit app (link above) + execution screenshots in submission PDF |

---

## Models Used

The pipeline runs on **Groq's free API tier** and uses two open-source LLMs from Meta's Llama family:

| Model | Size | Role |
|---|---|---|
| **Llama 3.1 8B Instant** (`llama-3.1-8b-instant`) | 8 billion parameters | Primary extraction model — handles most PDFs |
| **Llama 3.3 70B Versatile** (`llama-3.3-70b-versatile`) | 70 billion parameters | Fallback for large or complex documents |

### How the two models work together (groq-8b-focused mode)

The pipeline uses a **"try small first, fall back to big"** strategy:

1. **First attempt — 8B model.** For each PDF, the pipeline selects the most relevant text chunks using keyword scoring (looking for terms like the brand name, "prior authorization", "step therapy", etc.) and sends them to the smaller 8B model. This model is fast and handles most straightforward PA policies well.

2. **Fallback — 70B model.** If the PDF is too large to fit in the 8B model's context window, or if the document is unusually complex (e.g. multi-drug formularies with nested criteria), the pipeline automatically switches to the larger 70B model, which can process more text and handle nuanced policy language.

3. **Validation pass — 8B model.** After extraction, the pipeline optionally runs a second pass where the 8B model reviews the extracted values against the source text and corrects any mistakes. If the source text is too large for 8B, the 70B model handles validation instead.

4. **Regex pre-pass — no model.** For PDFs with a structured PA format (clear section headers like "Products Affected", "Prior Authorization Criteria"), the pipeline first tries deterministic regex extraction — no LLM needed. About 20% of PDFs are handled this way, saving API calls entirely.

In practice, the 8B model handles ~70% of extractions, the 70B model handles ~25%, and regex covers ~5% without any LLM call.

The submitted results were generated using **groq-8b-focused** mode.

---

## Access Score Framework

The Access Score quantifies how easy or difficult it is for a patient to get a specific biologic approved under a payer's PA policy. It ranges from **0** (most restrictive) to **100** (least restrictive), snapped to five discrete tiers: **0, 25, 50, 75, 100**.

### Scoring Components

The score is built from 10 weighted components derived from the 13 extracted parameters:

| Component | Weight | What it measures |
|---|---|---|
| Steps through Brands | 20 | Number of branded biologics the patient must try and fail first |
| Steps through Generic | 15 | Number of non-biologic drugs (e.g. methotrexate) required first |
| Phototherapy Required | 5 | Whether phototherapy is mandatory before biologic approval |
| TB Test Required | 5 | Whether a TB test is required before starting therapy |
| Age Restriction | 10 | Whether the policy restricts age beyond the FDA label |
| Initial Auth Duration | 15 | How long the initial approval lasts (longer = better) |
| Reauth Required + Duration | 10 | Whether reauthorization is needed and how often |
| Quantity Limits | 5 | Whether dosing/quantity restrictions exist |
| Specialist Requirement | 5 | Whether a specialist (e.g. dermatologist) must prescribe |
| Reauth Requirements | 10 | How strict the reauthorization criteria are (e.g. PASI scores vs. general improvement) |

**Total: 100 points**

### Scoring Logic (6 Layers)

**Layer 0 — Sparse Data Guard:** If fewer than 3 of 12 extractable parameters have real values, the extraction likely failed or the PDF doesn't cover this brand/indication. Scores conservatively: **0** if any step therapy was detected, **25** otherwise.

**Layer 1 — Hard Floors:** Extreme step therapy collapses directly to a tier:
- 3+ branded AND 2+ generic steps → **0**
- 5+ total steps → **0**
- 3+ branded steps alone → **25**

**Layer 2 — Step Therapy Base:** Steps are the dominant factor. The base score is set by total step count:
- 0 steps → base 90
- 1 generic step → base 50
- 1 branded step → base 40
- 2 steps (≤1 branded) → base 30
- 2 branded steps → base 20
- 3 steps → base 15, 4 steps → base 10

**Layer 3 — Secondary Adjustments:** Other parameters add or subtract from the base:
- Long auth duration (≥12 months): +5
- Short auth duration (<3 months): −5
- Frequent reauth (≤3 months): −8
- Phototherapy required: −8
- Specialist required: −5
- Restrictive reauth criteria (PASI/BSA thresholds): −5
- Age more restrictive than FDA label: −5

**Layer 4 — Interaction Penalty:** When 4+ restrictions are active simultaneously, the score is multiplied by a penalty factor (0.80 for 4, 0.70 for 5, 0.60 for 6+). This captures the compounding burden of stacked requirements.

**Layer 5 — Caps and Clamp:** Score is clamped to 0–100. If branded steps ≥ 2, score is capped at 50 regardless of other factors.

**Layer 5b — Baseline Offset:** A +15 point offset is applied to reflect that most policies do grant access eventually, even with restrictions. This shifts the score distribution upward to better use the full 0–100 range.

**Layer 6 — Bucket Snapping:** The continuous score is snapped to the nearest tier in {0, 25, 50, 75, 100}.

### Interpretation

| Score | Meaning |
|---|---|
| **100** | Minimal PA barriers — no step therapy, standard requirements |
| **75** | Low barriers — minor restrictions (e.g. TB test, specialist) |
| **50** | Moderate barriers — some step therapy or multiple restrictions |
| **25** | High barriers — significant step therapy or stacked restrictions |
| **0** | Very high barriers — extensive step therapy, near-impossible access |

---

## Dashboard Tabs

The Streamlit app has five tabs. Here's what each one does:

### Overview Tab
High-level summary with three charts (visible after ≥5 rows are processed): score distribution across tiers, average score per brand, and step therapy burden breakdown by brand.

### Results Tab
Filterable table of all extracted parameters. Select any row to inspect its values and see a score breakdown showing how each parameter contributes to the Access Score. Download the full dataset as CSV or Excel.

### PDF Explorer Tab
Deep-dive into any PDF — view page relevance scores, the exact text sent to the LLM at each filtering level, the full prompt with few-shot examples, and the raw extraction JSON.

### Run Tab
Execute the pipeline from the UI. Three modes: single PDF (pick from dropdowns), filtered subset (search by brand or file), or all rows. Live progress tracking, cache toggle, and validation skip.

### Guide Tab
Step-by-step walkthrough for new users — how to add API keys, run extractions, explore results, and tips for getting the best output.

---

## Running Locally

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env — add your Groq API key (free at console.groq.com)

# 3. Place data files
#    pipeline/data/PA_Business_Rules.xlsx   (submission template)
#    pipeline/data/pdfs/*.pdf               (policy PDFs)

# 4. Launch the UI
streamlit run app.py
```

**Output location:** After a full run, results are written to:
- `pipeline/output/results.csv`
- `pipeline/output/results.xlsx`

Both files contain identical data — one row per (Filename, Brand) pair with all 13 parameters and the Access Score.

### Running on Custom Data

To use the pipeline on your own set of PDFs:

**1. Prepare the Excel file**

Create an Excel file (`.xlsx`) following the same format as the submission template (`PA_Business_Rules.xlsx`). Open the **Submissions** tab — it should have at least two columns:
- **Filename** — the PDF filename (e.g. `my_policy.pdf`)
- **Brand** — the drug brand to extract for (e.g. `TREMFYA`)

Each row is one extraction task. The same PDF can appear multiple times with different brands if the policy covers multiple drugs.

**2. Place files**

```
pipeline/
  data/
    PA_Business_Rules.xlsx     # Your Excel (with a "Submissions" tab)
    pdfs/
      my_policy_1.pdf          # Your PDF files
      my_policy_2.pdf
      ...
```

**3. Run**

```bash
# Launch the UI — run extractions from the Run tab
streamlit run app.py

# Or use the CLI
python run_pipeline.py                                          # all rows
python run_pipeline.py --file my_policy_1.pdf --brand TREMFYA   # single row
python run_pipeline.py --subset 5                               # first 5 rows
```

Results are written to `pipeline/output/results.csv` and `pipeline/output/results.xlsx`.

---

## API Key Setup

Get a free Groq API key at [console.groq.com](https://console.groq.com).

```env
LLM_PROVIDER=groq-8b-focused
GROQ_API_KEYS=gsk_your_key_here

# Multiple keys for rate-limit rotation (recommended for faster processing):
# GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
```

---

## Output Format

Both `results.csv` and `results.xlsx` contain one row per (Filename, Brand) pair with these 15 columns:

| Column | Example Values |
|---|---|
| Filename | `330109-4880941.pdf` |
| Brand | `TREMFYA`, `STELARA`, etc. |
| Age | `No`, `FDA labelled age`, `>=18` |
| Step Therapy Requirements Documented in Policy | Free-text policy description |
| Number of Steps through Brands | `0`, `1`, `2`, `NA` |
| Number of Steps through Generic | `0`, `1`, `2`, `NA` |
| Step through-Phototherapy | `Yes`, `No`, `N/A` |
| TB Test required | `Y`, `No` |
| Quantity Limits | `NA`, specific limit text |
| Specialist Types | `Dermatologist`, `NA` |
| Initial Authorization Duration (in-months) | `6`, `12`, `24` |
| Reauthorization Duration (in-months) | `6`, `12`, `Unspecified` |
| Reauthorization Required | `Yes`, `No` |
| Reauthorization Requirements Documented in Policy | Free-text description |
| Access Score | `0`, `25`, `50`, `75`, `100` |

---

## Architecture

```
PDF files ──► pdf_extractor.py ──► extractor.py ──► validator.py ──► scorer.py ──► results.csv/xlsx
              (text + filter)      (LLM extract     (normalize +     (Access
                                    + validate)      biz rules)       Score)
```

**Stage 1–2: PDF Processing** (`pdf_extractor.py`) — Extracts text via PyMuPDF with 3-tier context filtering (page → paragraph → sentence). Classifies document type and detects universal criteria sections.

**Stage 3: LLM Extraction** (`extractor.py`, `prompts.py`) — 8b-first extraction with 70b fallback, regex pre-pass for structured PDFs, two-pass extract+validate, chunked extraction for large documents, multi-key rotation and response caching.

**Stage 4: Post-processing** (`scorer.py`, `validator.py`) — Value normalization, business rule checks, and Access Score computation using the 6-layer weighted model.

---

## Deployment (Streamlit Cloud)

1. Run the full pipeline locally to populate `pipeline/cache/` with all 79 rows
2. Commit `cache/`, `output/`, and `data/` to the repo
3. Push to GitHub
4. Go to [share.streamlit.io](https://share.streamlit.io) → point to your repo and `pipeline/app.py`
5. Add `GROQ_API_KEYS` as a secret in Settings → Secrets

**Cache on Streamlit Cloud:** The filesystem is ephemeral — files written at runtime are lost on restart. Pre-committed cache files are always available. New extractions done on the deployed app work within the session but won't persist across restarts. 

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — dashboard, run execution, PDF explorer |
| `run_pipeline.py` | CLI entry point for batch processing |
| `config.py` | Configuration, paths, brand lists |
| `pdf_extractor.py` | PDF text extraction and context filtering |
| `prompts.py` | LLM prompts and few-shot examples |
| `extractor.py` | LLM API client, caching, chunking, retry logic |
| `regex_extractor.py` | Deterministic regex extraction for structured PDFs |
| `scorer.py` | Access Score computation |
| `validator.py` | Output normalization and business rules |
| `normalizer.py` | Extraction output normalization |
| `step_analyzer.py` | Step therapy analysis |
| `debug.py` | Single-PDF deep inspection tool |
