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