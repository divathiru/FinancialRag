"""
streamlit_app.py — Stage 9: Streamlit front-end for the Cisco Earnings RAG system.

Layout
------
  Sidebar  : file uploader + Index button + indexing status panel
  Main     : question box (disabled until indexed) + answer area + history

Session state keys
------------------
  indexed       : bool  — True once at least one successful index run has finished
  history       : list  — Q&A records (newest-first display, appended FIFO)
  mistral_client: Mistral client, created once and reused
  index_summary : str   — human-readable result of the last index run

Pipeline
--------
  Uploaded PDFs → extract_pages() → chunk_pages() → embed_chunks() → store_chunks()
  Question      → retrieve()      → answer_with_sources()
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── path fix: allow imports from src/ ────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from extract   import extract_pages
from chunk     import chunk_pages
from embed     import embed_chunks, EMBEDDING_MODEL
from store     import store_chunks, make_prefixed_text, CHROMA_DIR, COLLECTION_NAME
from retrieve  import retrieve
from generate  import answer_with_sources, GENERATION_MODEL

# ── constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150
TOP_K         = 5

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cisco Earnings RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a3e, #0f3460);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Hero title */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: rgba(255,255,255,0.5);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Card-style containers */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg,
        rgba(167,139,250,0.1),
        rgba(96,165,250,0.1));
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.6rem;
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.7;
}

/* Source badge */
.src-badge {
    display: inline-block;
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.35);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    color: #34d399;
    margin: 3px 3px 3px 0;
    font-weight: 500;
}

/* Status success */
.status-ok {
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #34d399;
    font-size: 0.88rem;
    font-weight: 500;
}

/* Status warning */
.status-warn {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #fbbf24;
    font-size: 0.88rem;
}

/* History expander label */
.hist-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    font-style: italic;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

/* Text inputs */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.97rem !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 1.5px dashed rgba(167,139,250,0.4);
    border-radius: 12px;
    padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── session state init ────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "indexed":        False,
        "history":        [],      # list of {question, answer, sources, ts}
        "mistral_client": None,
        "index_summary":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Mistral client (lazy, cached in session) ──────────────────────────────────
def _get_client():
    if st.session_state.mistral_client is None:
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            st.error("❌ MISTRAL_API_KEY not found. Add it to your `.env` file and restart.")
            st.stop()
        try:
            from mistralai.client.sdk import Mistral
        except ImportError:
            st.error("❌ `mistralai` package not installed. Run `pip install -r requirements.txt`.")
            st.stop()
        st.session_state.mistral_client = Mistral(api_key=api_key)
    return st.session_state.mistral_client


# ── indexing pipeline ─────────────────────────────────────────────────────────
def _run_indexing(uploaded_files) -> dict:
    """
    Run extract → chunk → embed → store for a list of uploaded UploadedFile objects.
    Returns a summary dict: {files, pages, chunks, total_in_collection, elapsed}.
    """
    client = _get_client()

    t0 = time.time()
    all_pages: list[dict] = []

    # 1. Save uploads to a temp directory and extract pages
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        status = st.empty()

        for uf in uploaded_files:
            dest = tmp_path / uf.name
            dest.write_bytes(uf.read())
            status.info(f"📄 Extracting **{uf.name}** …")
            pages = extract_pages(dest)
            all_pages.extend(pages)

    n_files = len(uploaded_files)
    n_pages = len(all_pages)

    # 2. Chunk
    status.info(f"✂️  Chunking {n_pages} pages …")
    chunks = chunk_pages(all_pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    n_chunks = len(chunks)

    # 3. Embed (quarter-prefixed text — same approach as store.py)
    status.info(f"🔢 Embedding {n_chunks} chunks with `{EMBEDDING_MODEL}` …")
    prefixed_chunks = [{**c, "text": make_prefixed_text(c)} for c in chunks]
    vectors = embed_chunks(prefixed_chunks, client=client)

    # 4. Store
    status.info("💾 Upserting into ChromaDB …")
    total = store_chunks(chunks, vectors, persist_dir=CHROMA_DIR)

    elapsed = time.time() - t0
    status.empty()

    return {
        "files":   n_files,
        "pages":   n_pages,
        "chunks":  n_chunks,
        "total":   total,
        "elapsed": elapsed,
    }


# ── source rendering ──────────────────────────────────────────────────────────
def _render_sources(sources: list[dict]) -> str:
    """Return sources as HTML badge spans, grouped by file."""
    groups: dict[tuple, list[int]] = {}
    order:  list[tuple]            = []
    for s in sources:
        key = (s["quarter"], s["file"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s["page"])

    html_parts = []
    for key in order:
        quarter, fname = key
        pages = ", ".join(f"p.{p}" for p in groups[key])
        html_parts.append(
            f'<span class="src-badge">📄 {quarter} · {fname} · {pages}</span>'
        )
    return " ".join(html_parts)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📁 Index Documents")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.5);font-size:0.85rem;'>"
        "Upload one or more Cisco earnings PDFs, then click <strong>Index</strong>."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    do_index = st.button(
        "⚡ Index",
        disabled=(not uploaded),
        help="Extract, chunk, embed, and store the uploaded PDFs.",
    )

    # ── run indexing ──────────────────────────────────────────────────────────
    if do_index and uploaded:
        with st.spinner("Indexing — this may take ~10–30 s …"):
            try:
                summary = _run_indexing(uploaded)
                st.session_state.indexed       = True
                st.session_state.index_summary = summary
            except Exception as exc:
                st.error(f"❌ Indexing failed: {exc}")

    # ── indexing status panel ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Status")

    if st.session_state.indexed and st.session_state.index_summary:
        s = st.session_state.index_summary
        st.markdown(
            f"""<div class="status-ok">
            ✅ <strong>Index ready</strong><br>
            {s['files']} file(s) · {s['pages']} pages<br>
            {s['chunks']} new chunks upserted<br>
            {s['total']} total chunks in store<br>
            Done in {s['elapsed']:.1f}s
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-warn">⚠️ No index yet — upload PDFs and click Index.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f"<p style='color:rgba(255,255,255,0.3);font-size:0.75rem;'>"
        f"Embed model: <code>{EMBEDDING_MODEL}</code><br>"
        f"Chat model: <code>{GENERATION_MODEL}</code><br>"
        f"Collection: <code>{COLLECTION_NAME}</code><br>"
        f"Store: <code>{CHROMA_DIR}</code>"
        f"</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">📊 Cisco Earnings RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Ask questions over Cisco\'s FY25 quarterly earnings reports (Q1–Q3). '
    'Answers are grounded in the source documents — every figure is cited.</div>',
    unsafe_allow_html=True,
)

# ── question input ────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([5, 1], vertical_alignment="bottom")

with col_q:
    question = st.text_input(
        "Your question",
        placeholder=(
            "e.g. What was Cisco's total revenue for Q2 FY25?"
            if st.session_state.indexed
            else "Index documents first (sidebar →)"
        ),
        disabled=not st.session_state.indexed,
        label_visibility="collapsed",
        key="question_input",
    )

with col_btn:
    ask = st.button(
        "Ask →",
        disabled=(not st.session_state.indexed),
        use_container_width=True,
    )

# ── gate: show message when not indexed ──────────────────────────────────────
if not st.session_state.indexed:
    st.markdown(
        '<div class="status-warn" style="margin-top:0.8rem;">'
        '⚠️ <strong>Nothing indexed yet.</strong> '
        'Upload your PDFs in the sidebar and click <strong>⚡ Index</strong> before asking questions.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── answer ────────────────────────────────────────────────────────────────────
if ask and question.strip() and st.session_state.indexed:
    client = _get_client()

    with st.spinner("🔍 Retrieving relevant passages …"):
        chunks = retrieve(
            question.strip(),
            top_k=TOP_K,
            debug=False,
            mistral_client=client,
        )

    with st.spinner("🤖 Generating answer …"):
        result = answer_with_sources(
            question.strip(),
            chunks,
            client=client,
        )

    # Prepend to history (newest first)
    st.session_state.history.insert(0, {
        "question": question.strip(),
        "answer":   result["answer"],
        "sources":  result["sources"],
        "ts":       time.strftime("%H:%M:%S"),
    })

    # Rerun so the latest answer shows at the top of history below
    st.rerun()

# ── history list (newest on top) ──────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 💬 Q&A History")

    for i, record in enumerate(st.session_state.history):
        is_latest = (i == 0)
        label = f"**Q:** {record['question']}  ·  _{record['ts']}_"

        with st.expander(label, expanded=is_latest):
            st.markdown(
                f'<div class="answer-box">{record["answer"]}</div>',
                unsafe_allow_html=True,
            )

            if record["sources"]:
                st.markdown(
                    "<p style='margin-top:0.7rem;color:rgba(255,255,255,0.45);"
                    "font-size:0.8rem;font-weight:600;'>SOURCES</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _render_sources(record["sources"]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span style='color:rgba(255,255,255,0.3);font-size:0.8rem;'>"
                    "No sources available</span>",
                    unsafe_allow_html=True,
                )
