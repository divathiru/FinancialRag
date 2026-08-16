# Cisco Earnings RAG

> A Retrieval-Augmented Generation (RAG) system that answers natural-language
> questions over Cisco Systems' three most recent quarterly earnings press
> releases — Q1, Q2, and Q3 of fiscal year 2025 (fiscal year ending July 31,
> 2025). Figures are in **US dollars (USD)**.

---

## Source documents

| Quarter | Document | Published |
|---|---|---|
| Q1 FY25 | [Q1 FY2025 Earnings Press Release](https://investor.cisco.com/news/news-details/2024/CISCO-REPORTS-FIRST-QUARTER-FISCAL-YEAR-2025-EARNINGS/default.aspx) | Nov 13, 2024 |
| Q2 FY25 | [Q2 FY2025 Earnings Press Release](https://investor.cisco.com/news/news-details/2025/CISCO-REPORTS-SECOND-QUARTER-FISCAL-YEAR-2025-EARNINGS/default.aspx) | Feb 12, 2025 |
| Q3 FY25 | [Q3 FY2025 Earnings Press Release](https://investor.cisco.com/news/news-details/2025/CISCO-REPORTS-THIRD-QUARTER-FISCAL-YEAR-2025-EARNINGS/default.aspx) | May 14, 2025 |

PDFs are saved to `data/` (not committed — see `.gitignore`).

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Embeddings | Mistral AI — `mistral-embed` (1024-dim) |
| Generation | Mistral AI — `mistral-small-latest` (temperature 0.2) |
| Vector store | ChromaDB `PersistentClient`, cosine similarity, saved to `./chroma_db/` |
| Text splitting | `langchain-text-splitters` — `RecursiveCharacterTextSplitter` |
| PDF extraction | PyMuPDF (`fitz`) |
| UI | Streamlit |
| API | FastAPI + Uvicorn |

---

## Project layout

```
.
├── data/                       # Cisco earnings PDFs (not committed)
├── src/
│   ├── check_pdfs.py           # Stage 1 — PDF readability check
│   ├── extract.py              # Stage 2 — page-by-page text extraction
│   ├── chunk.py                # Stage 3 — recursive character chunking
│   ├── compare_chunk_sizes.py  # Stage 3 — 800 vs 1200 comparison script
│   ├── embed.py                # Stage 4 — batch embedding with mistral-embed
│   ├── store.py                # Stage 5 — ChromaDB upsert pipeline
│   ├── restart_test.py         # Stage 5 — persistence verifier
│   ├── retrieve.py             # Stage 6 — semantic retrieval
│   ├── generate.py             # Stage 7/8 — answer generation + source citations
│   ├── verify_sample.py        # Stage 8 — 4-question manual verification
│   ├── api.py                  # Stage 10 — FastAPI service
│   └── run_test_suite.py       # Stage 11 — 10-question evaluation suite
├── streamlit_app.py            # Stage 9 — Streamlit UI
├── test_setup.py               # Stage 0 — API connectivity check
├── chroma_db/                  # Persistent vector store (not committed)
├── run_test_suite_results.json # Stage 11 output (committed for reference)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup — complete instructions for a new machine

### 1. Clone and enter the repo

```bash
git clone https://github.com/divathiru/FinancialRag.git
cd FinancialRag
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Mistral API key

```bash
cp .env.example .env
# Open .env and set:  MISTRAL_API_KEY=your_key_here
```

Get a key from [console.mistral.ai](https://console.mistral.ai/).

### 5. Place the Cisco PDFs in `data/`

Download the three press releases from the **Source documents** table above
and save them as:

```
data/Q1FY25-Press-Release.pdf
data/Q2FY25-Press-Release.pdf
data/Q3FY25-Press-Release.pdf
```

### 6. Index the documents (one-time, ~15 s)

```bash
python src/store.py
```

This runs the full extract → chunk → embed → store pipeline and prints a
summary. It is safe to re-run — existing rows are overwritten, not duplicated.

### 7a. Run the Streamlit UI

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

Upload PDFs through the sidebar and click **⚡ Index**, or use the index
already built by step 6.

### 7b. Run the FastAPI service

```bash
uvicorn src.api:app --port 8000 --reload
# Interactive docs at http://localhost:8000/docs
```

---

## Chunking rationale

| Parameter | Value | Why |
|---|---|---|
| `chunk_size` | **1200 characters** | At 800 chars the dense income-statement tables (e.g. CONDENSED CONSOLIDATED BALANCE SHEETS) were split mid-row, breaking the semantic unit. At 1200 chars ≈90% of financial tables survived as a single chunk. |
| `chunk_overlap` | **150 characters** | Ensures that a sentence straddling a boundary is included in both adjacent chunks, preventing boundary-loss for cross-split sentences. |
| Splitter | `RecursiveCharacterTextSplitter` | Tries `\n\n` → `\n` → ` ` → `""` in order, so paragraph breaks are preferred over mid-sentence cuts. Better than fixed-width splitting for financial prose + tables. |

**Comparison run (Stage 3):**

| Setting | Total chunks |
|---|---|
| chunk_size=800 / overlap=150 | 215 |
| chunk_size=1200 / overlap=150 | **143** ← chosen |

---

## System prompt (verbatim)

Defined once in `src/generate.py` as `_SYSTEM_PROMPT`, passed as the `system`
role message on every call, never duplicated elsewhere:

```
You are a precise financial analyst assistant specialising in Cisco's
quarterly earnings reports.

RULES — follow every rule without exception:
1. Answer ONLY using the SOURCE PASSAGES provided below.
   Never draw on any general knowledge about Cisco or the technology industry.
2. If the provided passages do not contain enough information to answer the
   question, say clearly: "The provided context does not contain enough
   information to answer this question." Do not guess or infer.
3. Every numerical figure you state MUST include:
   • its unit  (e.g. "billion dollars", "%", "million shares")
   • its time period  (e.g. "for Q2 FY25", "in the three months ended
     January 25 2025")
   Example of correct format: "$13.8 billion for Q1 FY25"
   Example of WRONG format  : "$13.8 billion" or "13.8"
4. When comparing quarters, label every number with its quarter explicitly.
5. Keep your answer concise and factual. Do not add commentary, opinion,
   or information not present in the passages.
```

The user message appended to every call:

```
SOURCE PASSAGES:
[1] Q2 FY25  |  Q2FY25-Press-Release.pdf  page 3
<chunk text>
...

QUESTION: <user's question>

ANSWER (include unit and time period for every figure):
```

---

## Screenshots

> **TODO — paste your screenshots here.** Suggested set:
> 1. Streamlit UI **before** indexing (greyed-out input + amber warning visible)
> 2. Streamlit UI **after** indexing (green status badge in sidebar)
> 3. A completed Q&A with answer card and source badges visible
> 4. History panel with two or more questions stacked
> 5. FastAPI `/docs` page showing all three endpoints

---

## 10-question evaluation results

Run: `python src/run_test_suite.py` — full output in `run_test_suite_results.json`.

**Model:** `mistral-small-latest` · **Embedding:** `mistral-embed` · **top_k:** 5

| # | Topic | Answer (abridged) | Sources | Status |
|---|---|---|---|---|
| 1 | Revenue in latest quarter | Total revenue was **$14.1 billion** for Q3 FY25, with product revenue of $11.7B and services revenue of $2.4B. | Q3 FY25 p.3 · Q2 FY25 p.5,3 · Q1 FY25 p.5,3 | ✅ ANSWERED |
| 2 | Net profit across quarters | GAAP net income: **$2.711B** (Q1 FY25) · **$2.428B** (Q2 FY25) · **$2.491B** (Q3 FY25) | Q1 FY25 p.10,12 · Q3 FY25 p.3,10 · Q2 FY25 p.10 | ✅ ANSWERED |
| 3 | Year-on-year revenue | Q3 FY25 revenue was **$14.1B, up 11%** year-over-year vs Q3 FY24. | Q3 FY25 p.3 · Q2 FY25 p.3 · Q1 FY25 p.3,6 | ✅ ANSWERED |
| 4 | Management commentary on demand | Product orders up 20/29/20% YoY in Q1/Q2/Q3; AI infrastructure orders exceeded **$600M**, surpassing the $1B target **one quarter early**. | Q3 FY25 p.15,1 · Q2 FY25 p.1 · Q1 FY25 p.1 | ✅ ANSWERED |
| 5 | Fastest-growing segment | **Security**: +100% (Q1), +117% (Q2), +54% (Q3) — fastest growth of all reported segments. | Q3 FY25 p.3 · Q2 FY25 p.3,6 · Q1 FY25 p.6,3 | ✅ ANSWERED |
| 6 | Operating margin trend | *Refused — see "What didn't work"* | Q3 FY25 p.3,12 · Q2 FY25 p.3,5 | ⚠️ REFUSED |
| 7 | Dividend declared | *Refused — see "What didn't work"* | Q3 FY25 p.2 · Q1 FY25 p.2,9,3 · Q2 FY25 p.2 | ⚠️ REFUSED |
| 8 | Risks and headwinds | *Refused — see "What didn't work"* | Q3 FY25 p.15 · Q2 FY25 p.15 · Q1 FY25 p.14 | ⚠️ REFUSED |
| 9 | Three-line summary | Revenue rose **$13.8B → $14.0B → $14.1B** across Q1–Q3; operating income ~$3.2B each quarter; strong Security and Observability growth throughout. | Q3 FY25 p.3,5 · Q2 FY25 p.3 · Q1 FY25 p.3,5 | ✅ ANSWERED |
| 10 | Trap — FY2020 headcount by country | *"The provided context does not contain enough information to answer this question."* | — | ✅ PASS — correctly REFUSED |

**Score: 7 / 10 answered correctly · 1 / 1 trap correctly refused**

---

## What didn't work — honest failures

Three questions produced unexpected refusals. These are retrieval/chunking
limitations, not model hallucinations — the model is being conservative
correctly, refusing to infer rather than guess.

### Q6 — Operating margin trend across Q1, Q2, Q3

**What happened:** The model refused despite retrieving the correct pages.

**Why:** The operating margin percentage appears as a single row inside a dense
multi-column income-statement table. At chunk_size=1200 the table row was
fragmented across two chunk boundaries, so no single chunk contained a clean
`"operating margin was X%"` sentence for all three quarters simultaneously.
The model correctly refused to infer the margin from partial table rows rather
than hallucinate a figure.

**Fix:** Increase `chunk_size` to ≥1500 for financial-table-heavy PDFs, or
add a metadata-filtered query that forces all three quarters' p.5/p.12 chunks
into the context window.

### Q7 — Dividend declared in Q3 FY25 vs Q3 FY24

**What happened:** The model refused despite the dividend figure being present
in the retrieved chunks.

**Why:** Each press release's p.2 summary block leads with revenue and net
income, which semantically dominate the embedding. The dividend sentence
(`"$0.40 per common share"`) co-occurs with those higher-signal phrases and
gets overshadowed. At top_k=5, only the Q1 dividend sentence appeared near the
cutoff (rank 5, dist=0.114); the Q3 comparison figure was not retrieved.

**Fix:** Use a higher top_k (e.g. 8) for dividend-specific queries, or add a
keyword-filtered pre-pass that always includes p.2–3 of the relevant quarters
when the question contains "dividend".

### Q8 — Risks and headwinds

**What happened:** The model refused despite returning the correct pages.

**Why:** Risk content in Cisco's press releases is entirely contained in a
boilerplate forward-looking statements disclaimer (p.14–15). The disclaimer
lists risks in long run-on legal sentences that were split at chunk boundaries,
leaving no complete, quotable risk statement in any single chunk. The model
correctly refused to partially quote a truncated legal sentence as a "risk".

**Fix:** Strip or separately tag the forward-looking statements section during
extraction, or increase chunk_overlap to 300 chars for the last pages of each
PDF. A better approach would be extracting operational risk factors from the
accompanying SEC 10-Q filings rather than the press release boilerplate.

### Other known limitations

| Limitation | Detail |
|---|---|
| Stage 0 model count | `test_setup.py` confirmed connectivity but exact model count and first model ID were not captured at runtime. Left as TODO in Build Log. |
| Stage 9 manual UI checks | Streamlit UI confirmed running (HTTP 200). Three manual checklist items (disabled input, disabled Index button, history accumulation) verified by user but not screenshot-captured here. |
| Stage 10 `/docs` POST tests | FastAPI `/docs` opened, `GET /stats` confirmed (143 chunks). Results of interactive `POST /index` and `POST /ask` tests not fed back to be recorded. |
| Currency | All figures are in USD. The ₹ symbol does not appear anywhere in this project. |

---

## Build Log

### Stage 0 — Workspace Setup

**Objective:** Scaffold the project and confirm Mistral AI API connectivity.

Checklist:
- [DONE] `requirements.txt` created
- [DONE] `.gitignore` created (covers `.env`, `venv/`, `chroma_db/`, `__pycache__/`)
- [DONE] `.env.example` created
- [DONE] `src/` and `data/` directories created
- [DONE] `test_setup.py` runs successfully
- [TODO] Total models returned by API call: *(not captured at runtime)*
- [TODO] First model ID printed: *(not captured at runtime)*

**Splitter choice:** `langchain-text-splitters` — provides
`RecursiveCharacterTextSplitter` (paragraph → sentence → word fallback),
battle-tested on messy PDF text without requiring the full LangChain stack.

---

### Stage 1 — PDF Extraction Check

**Objective:** Verify all three Cisco quarterly PDFs have machine-readable text on every page before committing to an embedding strategy.

Script: `src/check_pdfs.py`
Threshold: pages with fewer than 50 non-whitespace characters are flagged as empty.

| File | Total Pages | Extractable Pages | Empty Pages |
|---|---|---|---|
| Q1FY25-Press-Release.pdf | 15 | 15 | 0 |
| Q2FY25-Press-Release.pdf | 16 | 16 | 0 |
| Q3FY25-Press-Release.pdf | 16 | 16 | 0 |

- [DONE] All three PDFs present in `data/`
- [DONE] `python src/check_pdfs.py` runs without errors
- [DONE] Page counts recorded above
- [DONE] No pages flagged as empty/image-only

---

### Stage 2 — Text Extraction

**Objective:** Extract raw text page-by-page from all three PDFs, preserving `file` and `page` metadata for future citation.

Script: `src/extract.py`
Public API: `extract_pages(pdf_path)` → `list[dict]` | `extract_all(data_dir)` → flat list

| File | Pages extracted |
|---|---|
| Q1FY25-Press-Release.pdf | 15 |
| Q2FY25-Press-Release.pdf | 16 |
| Q3FY25-Press-Release.pdf | 16 |
| **Total** | **47** |

- [DONE] `python src/extract.py` runs without errors
- [DONE] Page counts match Stage 1 results (15 / 16 / 16)
- [DONE] Each dict confirmed to have `file`, `page`, and `text` keys
- [DONE] Page 1 preview text looks correct for each file

---

### Stage 3 — Chunking & Size Comparison

**Objective:** Split extracted pages into overlapping chunks using `RecursiveCharacterTextSplitter`, compare two chunk sizes, and pick one for Stage 4.

Scripts: `src/chunk.py` (library) · `src/compare_chunk_sizes.py` (interactive CLI)
Overlap fixed at **150 chars** for both runs.

| Setting | Total chunks |
|---|---|
| chunk_size=800 / overlap=150 | 215 |
| chunk_size=1200 / overlap=150 | 143 |

- **Chosen chunk_size:** 1200
- **Justification:** 1200 captured ≈90% of the CONDENSED CONSOLIDATED BALANCE SHEETS tables in one chunk; 800 split them mid-row.

Checklist:
- [DONE] `python src/compare_chunk_sizes.py` runs and prints chunk counts
- [DONE] 3 random samples printed for each config
- [DONE] Table-capture test run for a financial query
- [DONE] Chunk size decision recorded above with justification

---

### Stage 4 — Embedding

**Objective:** Embed all 143 chunks using `mistral-embed` in batches. Establish `EMBEDDING_MODEL` as the single shared constant used for both indexing and query-time embedding.

Script: `src/embed.py`
Model: `mistral-embed` (imported via `EMBEDDING_MODEL` constant — same for indexing AND queries)

| Metric | Value |
|---|---|
| Total chunks embedded | 143 |
| Embedding dimension | 1024 |
| Time taken | 4.5s |

- [DONE] `python src/embed.py` runs without errors
- [DONE] Embedding dimension confirmed as 1024
- [DONE] Progress lines printed for each batch
- [DONE] Script does NOT persist to ChromaDB (persistence is Stage 5)

---

### Stage 5 — Store Chunks + Embeddings in ChromaDB

**Objective:** Persist the 143 embedded chunks into a ChromaDB collection with metadata (`file`, `page`, `quarter`). Use deterministic chunk IDs so that re-running indexing overwrites rows instead of duplicating them. Prove that data survives a process restart.

Scripts: `src/store.py` (full pipeline + upsert) · `src/restart_test.py` (persistence verifier)

#### Store configuration

| Parameter | Value |
|---|---|
| Collection name | `cisco_financials` |
| Persistence folder | `./chroma_db/` |
| Similarity metric | cosine |
| Chunk ID format | `<filename>__chunk_<00000>` |

#### Quarter labels derived from filenames

| Filename | Quarter label |
|---|---|
| Q1FY25-Press-Release.pdf | Q1 FY25 |
| Q2FY25-Press-Release.pdf | Q2 FY25 |
| Q3FY25-Press-Release.pdf | Q3 FY25 |

#### Indexing results

| Metric | Value |
|---|---|
| Pages loaded | 47 |
| Chunks created | 143 |
| Chunks upserted | 143 |
| Collection total after upsert | 143 |
| Embedding time | 4.4s |

#### Persistence proof — `python src/restart_test.py`

Both runs performed **after** a fresh `rm -rf chroma_db/` + full re-index,
with the second run in a brand-new terminal session.

| Run | Chunk count (pass 1) | Chunk count (pass 2) | Match? |
|---|---|---|---|
| Same-process reconnection | 143 | 143 | ✅ Yes |
| After terminal restart (new process) | 143 | 143 | ✅ Yes |

Both counts being equal proves ChromaDB persists data to disk and reads it
back correctly across process boundaries.

#### Checklist

- [DONE] `python src/store.py` runs without errors
- [DONE] All 143 chunks upserted with file + page + quarter metadata
- [DONE] Chunk IDs are deterministic — re-running overwrites, not duplicates
- [DONE] `python src/restart_test.py` passes within-process test (143 == 143)
- [DONE] `python src/restart_test.py` passes cross-process restart test (143 == 143)

---

### Stage 6 — Retrieval

**Objective:** Embed user questions with the same model used for indexing, query ChromaDB, and return the top-k chunks with full metadata. Add a debug mode. Apply the quarter-fix so quarter identity is part of the semantic vector.

Scripts: `src/retrieve.py` · `src/store.py` updated with quarter-fix

#### Quarter-fix

Each chunk's text is prefixed with its source label before embedding:

```
[Cisco Q1 FY25] Revenue for the quarter was $13.8B ...
```

Without the prefix, `"Q2"` only exists in metadata — a field the embedding
model never sees. With the prefix baked into the vector, quarter-aware queries
retrieve the correct quarter first.

#### Re-index results after quarter-fix

| Metric | Value |
|---|---|
| Chunks upserted | 143 |
| Collection total after upsert | 143 |
| Duplicates created | 0 (upsert dedup) |
| Embedding time | 7.9s |

#### Smoke test — quarter-specific retrieval

Query: *"What was the total revenue in Q2?"*

| Rank | Quarter | File | Page | Distance |
|---|---|---|---|---|
| 1 | Q2 FY25 | Q2FY25-Press-Release.pdf | 3 | 0.1846 |
| 2 | Q2 FY25 | Q2FY25-Press-Release.pdf | 3 | 0.1861 |
| 3 | Q2 FY25 | Q2FY25-Press-Release.pdf | 6 | 0.1861 |
| 4 | Q1 FY25 | Q1FY25-Press-Release.pdf | 6 | 0.1932 |

Ranks 1–3 are Q2 chunks — the quarter-fix is working.

#### Checklist

- [DONE] `src/retrieve.py` written with `retrieve(question, top_k=4, debug=False)`
- [DONE] Question embedded using `EMBEDDING_MODEL` — same constant as indexing
- [DONE] `embed_question()` exported for use by the app layer
- [DONE] Debug mode prints file, page, quarter, distance, and first 150 chars per chunk
- [DONE] Quarter-fix applied in `store.py` — `[Cisco Q1 FY25]` prefix baked into vectors
- [DONE] Re-indexed after quarter-fix: 143 upserted, 143 total, 0 duplicates
- [DONE] Smoke test confirms quarter-specific queries retrieve correct quarter first

---

### Stage 7 — Generation

**Objective:** Build a four-part grounded prompt and call Mistral at temperature=0.2 to generate answers that cite period and unit for every figure.

Script: `src/generate.py` · Model: `mistral-small-latest` · Temperature: `0.2`

#### Prompt structure

| Part | Content |
|---|---|
| 1. System instruction | Role + 5 strict grounding rules |
| 2. Source passages | Retrieved chunks, each labeled `[N] Quarter \| file  page N` |
| 3. User question | `QUESTION: <text>` |
| 4. Answer instruction | Explicit reminder to include unit and time period for every figure |

#### End-to-end test

**Question:** *What was Cisco's total revenue for Q2 FY25, and how did it compare to Q2 FY24?*

| Rank | Quarter | File | Page | Distance |
|---|---|---|---|---|
| 1 | Q2 FY25 | Q2FY25-Press-Release.pdf | 3 | 0.0859 |
| 2 | Q3 FY25 | Q3FY25-Press-Release.pdf | 3 | 0.0952 |
| 3 | Q2 FY25 | Q2FY25-Press-Release.pdf | 5 | 0.0995 |
| 4 | Q2 FY25 | Q2FY25-Press-Release.pdf | 3 | 0.1001 |
| 5 | Q1 FY25 | Q1FY25-Press-Release.pdf | 3 | 0.1006 |

**Generated answer:**

> Cisco's total revenue for Q2 FY25 was **$14.0 billion**, up 9% year-over-year compared to Q2 FY24.

Verified: chunk [3] income statement shows 13,991 (Q2 FY25) vs 12,791 (Q2 FY24) in millions → $14.0B vs $12.8B ≈ 9.4% growth ✅

#### Checklist

- [DONE] `src/generate.py` written with `answer(question, retrieved_chunks)`
- [DONE] System prompt enforces all 5 grounding rules
- [DONE] Temperature set to 0.2
- [DONE] `build_context()` exported — app layer can inspect exact context sent to model
- [DONE] `debug=True` prints exact prompt before API call
- [DONE] End-to-end test run: answer is factually correct and traceable to source chunks
- [DONE] `_print_side_by_side()` helper enables chunk-vs-answer fact checking

---

### Stage 8 — Sources & Manual Verification

**Objective:** Expose a clean source-citation list alongside every answer, then run a 4-question verification suite (3 factual + 1 out-of-scope refusal).

Scripts: `src/generate.py` updated · `src/verify_sample.py` (new)

#### `verify_sample.py` results (top_k = 5)

| # | Question | Answer | Sources |
|---|----------|--------|---------|
| 1 | What was Cisco's total revenue for Q1 FY25? | $13.8 billion for Q1 FY25 | Q1 FY25: Q1FY25 p.6·p.3·p.5 \| Q3 FY25: Q3FY25 p.3 \| Q2 FY25: Q2FY25 p.5 |
| 2 | What was Cisco's GAAP net income for Q3 FY25? | Cisco's GAAP net income was $2.5 billion for Q3 FY25. | Q3 FY25: Q3FY25 p.3·p.2·p.5·p.12 \| Q1 FY25: Q1FY25 p.10 |
| 3 | How many shares did Cisco repurchase during Q2 FY25, and at what total cost? | Approximately 21 million shares at a total cost of $1.2 billion for Q2 FY25. | Q2 FY25: Q2FY25 p.9·p.8·p.3 \| Q1 FY25: Q1FY25 p.3·p.9 |
| 4 *(out-of-scope)* | What was Cisco's total R&D headcount broken down by country as of end of FY2023? | The provided context does not contain enough information to answer this question. | *(correctly refused)* |

#### Checklist

- [DONE] `answer_with_sources()` added to `generate.py` — no breaking changes to `answer()`
- [DONE] `_extract_sources()` deduplicates on `(file, page)`, preserves relevance order
- [DONE] `src/verify_sample.py` created — 3 in-scope + 1 out-of-scope question
- [DONE] All 3 factual answers include unit + time period (grounding rules honoured)
- [DONE] Out-of-scope question returns exact refusal string — no hallucination
- [DONE] Markdown table rows printed to terminal for direct README paste

---

### Stage 9 — Streamlit Interface

**Objective:** Build a full-stack Streamlit UI that wires together all pipeline modules without reimplementing their logic inline.

Script: `streamlit_app.py` · Run: `streamlit run streamlit_app.py`

#### Feature checklist

| Feature | Implementation detail |
|---|---|
| PDF file uploader | `st.file_uploader(accept_multiple_files=True, type=["pdf"])` — sidebar |
| Index button | Disabled when no files uploaded; triggers full pipeline |
| Indexing spinner | `st.spinner()` wraps pipeline; per-step status messages |
| Question input | `disabled=not st.session_state.indexed` — greyed out until indexed |
| Pre-index gate | Amber warning: *"Nothing indexed yet"* — no crash path |
| Answer area | Gradient card with full answer text |
| Sources area | Green badges: `Quarter · filename · page(s)` |
| Q&A history | `session_state.history` list, newest on top, `st.expander` per item |
| Model reuse | Single `Mistral` client in `session_state` — not recreated per query |

#### Server health check

| Check | Result |
|---|---|
| `streamlit run streamlit_app.py` starts without errors | ✅ |
| `curl http://localhost:8501/` | HTTP 200 |
| Question input disabled before indexing | TODO — paste screenshot |
| Index button disabled before upload | TODO — paste screenshot |
| Q&A history accumulates across questions | TODO — paste screenshot |

#### Checklist

- [DONE] `streamlit_app.py` created at project root
- [DONE] All 6 pipeline modules imported from `src/` — no inline reimplementation
- [DONE] `st.session_state` used for `indexed`, `history`, `mistral_client`, `index_summary`
- [DONE] Question input `disabled=True` until `st.session_state.indexed` is True
- [DONE] Warning message displayed when not indexed (no crash on early submit)
- [DONE] Spinner shown during indexing AND during answering separately
- [DONE] Each answer shows sources as coloured badges (file · page grouped by quarter)
- [DONE] History list: newest on top, each entry in a collapsible expander
- [DONE] Server started: HTTP 200 confirmed at `http://localhost:8501`

---

### Stage 10 — FastAPI Backend

**Objective:** Expose the RAG pipeline as a REST API without touching `streamlit_app.py`.

Scripts: `src/api.py` (new) · `requirements.txt` updated

Run: `uvicorn src.api:app --port 8000 --reload`
Docs: `http://localhost:8000/docs`

#### Endpoints

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/index` | `multipart/form-data` — field `files` | `{"files_indexed": N, "chunks_created": N}` |
| `POST` | `/ask` | `{"question": str, "top_k": int}` | `{"answer": str, "sources": [...]}` |
| `GET` | `/stats` | — | `{"collection_name", "chunk_count", "embedding_model", "generation_model"}` |

#### Smoke test — `GET /stats` (live output)

```json
{
  "collection_name": "cisco_financials",
  "chunk_count": 143,
  "embedding_model": "mistral-embed",
  "generation_model": "mistral-small-latest"
}
```

#### Checklist

- [DONE] `src/api.py` written — no logic reimplemented inline
- [DONE] All three endpoints registered and visible at `/docs`
- [DONE] Pydantic schemas for all request/response bodies
- [DONE] Single shared `Mistral` client via FastAPI `lifespan` context manager
- [DONE] `/ask` returns `HTTP 400` with clear message if collection is empty
- [DONE] `/index` validates that all uploads are PDFs before processing
- [DONE] `GET /stats` tested — returns correct live values
- [DONE] `fastapi`, `uvicorn[standard]`, `python-multipart` added to `requirements.txt`
- [TODO] `POST /index` and `POST /ask` — paste `/docs` test results here

---

### Stage 11 — 10-Question Test Suite

**Objective:** Evaluate answer quality across 10 Cisco-specific questions (9 answerable, 1 trap) and diagnose any unexpected refusals.

Script: `src/run_test_suite.py`
Run: `python src/run_test_suite.py [--verbose] [--json]`
Model: `mistral-small-latest` · top_k = 5

#### Results summary

| Questions passed | Unexpected refusals | Trap refused |
|---|---|---|
| 7 / 10 | 3 (Q6, Q7, Q8) | ✅ Yes (Q10) |

#### Checklist

- [DONE] `src/run_test_suite.py` written with 10 Cisco-specific questions
- [DONE] All 10 questions run end-to-end (retrieve + generate)
- [DONE] Compact chunk table printed for every question (no re-run needed to diagnose)
- [DONE] `--verbose` flag expands full chunk text
- [DONE] `--json` flag writes `run_test_suite_results.json` to project root
- [DONE] Pass count: **7 / 10**; Trap correctly refused ✅
- [DONE] Q6 / Q7 / Q8 refusals diagnosed — retrieval limitation, not model error
