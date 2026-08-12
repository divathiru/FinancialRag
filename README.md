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
- [ ] `requirements.txt` created
- [ ] `.gitignore` created (covers `.env`, `venv/`, `chroma_db/`, `__pycache__/`)
- [ ] `.env.example` created
- [ ] `src/` and `data/` directories created
- [ ] `test_setup.py` runs successfully
- [ ] Total models returned by API call: TODO
- [ ] First model ID printed: TODO

**Splitter choice:** `langchain-text-splitters` — provides
`RecursiveCharacterTextSplitter` (paragraph → sentence → word fallback),
battle-tested on messy PDF text without requiring the full LangChain stack.
