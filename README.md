# Cisco Earnings RAG

A Retrieval-Augmented Generation (RAG) system that answers questions over
Cisco's FY2025 quarterly earnings press releases (Q1–Q3, fiscal year ending
July 31).

**Tech stack:** Python 3.10+ · OpenAI (GPT-4o + text-embedding-3-small) ·
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
├── .env.example        # Copy to .env and fill in OPENAI_API_KEY
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
# Edit .env and set OPENAI_API_KEY=sk-...

# 4. Run the Stage 0 connectivity check
python test_setup.py
```

---

## Build Log

### Stage 0 — Workspace Setup

**Objective:** Scaffold the project and confirm OpenAI API connectivity.

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

