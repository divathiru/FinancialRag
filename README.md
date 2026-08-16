# Cisco Earnings RAG

A Retrieval-Augmented Generation (RAG) system that answers questions over
Cisco's FY2025 quarterly earnings press releases (Q1–Q3, fiscal year ending
July 31).

**Tech stack:** Python 3.10+ · Mistral AI (mistral-embed) ·
ChromaDB · Streamlit

---

## Project Layout

```
.
├── data/               # Cisco earnings PDFs (not committed)
├── src/                # Application source code
├── chroma_db/          # Persistent vector store (not committed, regenerated)
├── test_setup.py       # Stage 0 connectivity check
├── requirements.txt
├── .env.example        # Copy to .env and fill in MISTRAL_API_KEY
└── README.md
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Edit .env and set MISTRAL_API_KEY=...

# 4. Run the Stage 0 connectivity check
python test_setup.py
```

---

## Build Log

### Stage 0 — Workspace Setup

**Objective:** Scaffold the project and confirm Mistral AI API connectivity.

Checklist:
- [] `requirements.txt` created
- [] `.gitignore` created (covers `.env`, `venv/`, `chroma_db/`, `__pycache__/`)
- [] `.env.example` created
- [] `src/` and `data/` directories created
- [] `test_setup.py` runs successfully
- [] Total models returned by API call: TODO
- [] First model ID printed: TODO

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
| Cisco_Q1_FY26.pdf | 15 | 15 | 0 |
| Cisco_Q2_FY26.pdf | 16 | 16 | 0 |
| Cisco_Q3_FY26.pdf | 16 | 16 | 0 |

- [DONE] All three PDFs present in `data/`
- [DONE] `python src/check_pdfs.py` runs without errors
- [DONE] Page counts recorded above
- [DONE] No pages flagged as empty/image-only (or OCR plan noted if any are)
<img width="801" height="550" alt="Screenshot from 2026-08-13 08-45-17" src="https://github.com/user-attachments/assets/a5a32f25-3edc-4c12-ad61-00860c1b8d47" />
<img width="805" height="429" alt="Screenshot from 2026-08-13 08-45-35" src="https://github.com/user-attachments/assets/37ba09da-1ec6-477a-9f80-12e03ea86a13" />
<img width="800" height="486" alt="Screenshot from 2026-08-13 08-45-49" src="https://github.com/user-attachments/assets/7ae96a8f-e2b5-44e9-844b-ba3d4af44e7f" />


---

### Stage 2 — Text Extraction

**Objective:** Extract raw text page-by-page from all three PDFs, preserving `file` and `page` metadata for future citation.

Script: `src/extract.py`  
Public API: `extract_pages(pdf_path)` → `list[dict]` | `extract_all(data_dir)` → flat list

| File | Pages extracted |
|---|---|
| Cisco_Q1_FY26.pdf | 15 |
| Cisco_Q2_FY26.pdf | 16 |
| Cisco_Q3_FY26.pdf | 16 |
| **Total** | **47** |

- [DONE] `python src/extract.py` runs without errors
- [DONE] Page counts match Stage 1 results (15 / 16 / 16)
- [DONE] Each dict confirmed to have `file`, `page`, and `text` keys
- [DONE] Page 1 preview text looks correct for each file
![alt text](<Screenshot from 2026-08-13 09-03-01.png>) ![alt text](<Screenshot from 2026-08-13 09-03-16.png>) ![alt text](<Screenshot from 2026-08-13 09-03-36.png>)

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
- **Justification:** 1200 captured the 90% of the CONDENSED CONSOLIDATED BALANCE SHEETS tables in one chunk; 800 split it

Checklist:
- [Done] `python src/compare_chunk_sizes.py` runs and prints chunk counts
- [Done] 3 random samples printed for each config
- [Done] Table-capture test run for a financial query
- [Done] Chunk size decision recorded above with justification

![alt text](<Screenshot from 2026-08-13 09-25-39.png>) ![alt text](<Screenshot from 2026-08-13 09-25-52.png>)

---

### Stage 4 — Embedding

**Objective:** Embed all 143 chunks using `mistral-embed` in batches. Establish `EMBEDDING_MODEL` as the single shared constant used for both indexing and query-time embedding.

Script: `src/embed.py`  
Model: `mistral-embed` (imported via `EMBEDDING_MODEL` constant — same for indexing AND queries)

| Metric | Value |
|---|---|
| Total chunks embedded | 143 |
| Embedding dimension | 1024 |
| Time taken (s) | 4.5s |
    
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

Both runs performed **after** a fresh `rm -rf chroma_db/` + full re-index, with the second run in a brand-new terminal session.

| Run | Chunk count (pass 1) | Chunk count (pass 2) | Match? |
|---|---|---|---|
| Same-process reconnection | 143 | 143 | ✅ Yes |
| After terminal restart (new process) | 143 | 143 | ✅ Yes |

Both counts are written here explicitly — the two numbers being equal is the proof that ChromaDB persists data to disk and reads it back correctly across process boundaries.

#### Checklist

- [DONE] `python src/store.py` runs without errors
- [DONE] All 143 chunks upserted with file + page + quarter metadata
- [DONE] Chunk IDs are deterministic — re-running overwrites, not duplicates
- [DONE] `python src/restart_test.py` passes within-process test (143 == 143)
- [DONE] `python src/restart_test.py` passes cross-process restart test (143 == 143)

---

### Stage 6 — Retrieval

**Objective:** Embed user questions with the same model used for indexing, query ChromaDB, and return the top-k chunks with full metadata. Add a debug mode that makes retrieval quality visible before any generated answer is trusted. Apply the quarter-fix so quarter identity is part of the semantic vector.

Scripts: `src/retrieve.py` (retrieval module + interactive CLI) · `src/store.py` updated with quarter-fix

#### Quarter-fix (applied to store.py before re-indexing)

Each chunk's text is prefixed with its source label before embedding:

```
[Cisco Q1 FY25] Revenue for the quarter was $13.8B ...
```

**Why this matters:** without the prefix, `"Q2"` only exists in the `metadata` dict — a field the embedding model never sees. A question like *"What was Q2 revenue?"* had no semantic signal to distinguish Q1 from Q2 chunks. With the prefix baked into the vector, quarter-aware queries retrieve the correct quarter first.

The same prefixed text is stored as the ChromaDB document, so the LLM also receives the quarter label in its context window.

#### Re-index results after quarter-fix

| Metric | Value |
|---|---|
| Chunks upserted | 143 |
| Collection total after upsert | 143 |
| Duplicates created | 0 (upsert dedup) |
| Embedding time | 7.9s |

#### Retrieval design

| Parameter | Value |
|---|---|
| Embedding model | `mistral-embed` (same constant as indexing) |
| Similarity metric | cosine distance |
| Default top_k | 4 |
| Debug mode | `retrieve(..., debug=True)` |

#### Debug output format (every retrieved chunk)

```
[1] Q2 FY25  |  Q2FY25-Press-Release.pdf  p.3  |  dist=0.1846
     [Cisco Q2 FY25] Financial Summary ... Revenue ...
```

Fields printed: rank · quarter · filename · page · cosine distance · first 150 chars of text.

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