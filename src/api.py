"""
src/api.py — Stage 10: FastAPI service for the Cisco Earnings RAG system.

Endpoints
---------
POST /index
    Upload one or more PDF files.  Runs extract → chunk → embed → store and
    returns the number of files indexed and chunks created.
    Request : multipart/form-data, field "files" (one or more PDF uploads)
    Response: {"files_indexed": int, "chunks_created": int}

POST /ask
    Ask a question over the indexed documents.
    Request : JSON {"question": str, "top_k": int (default 5)}
    Response: {"answer": str, "sources": [{"file", "page", "quarter"}, ...]}

GET /stats
    Return a snapshot of the current index state and model configuration.
    Response: {
        "collection_name"  : str,
        "chunk_count"      : int,
        "embedding_model"  : str,
        "generation_model" : str,
    }

Running
-------
    source venv/bin/activate
    uvicorn src.api:app --reload --port 8000

    Interactive docs: http://localhost:8000/docs
    OpenAPI JSON    : http://localhost:8000/openapi.json

Design notes
------------
- A single Mistral client is created at startup via FastAPI's lifespan context
  manager and stored in app.state.  Every request reuses it — no per-request
  authentication overhead.
- All pipeline logic is imported from the existing src/ modules; nothing is
  reimplemented inline.
- The API layer is intentionally stateless beyond the on-disk ChromaDB store:
  POST /index is idempotent (upsert), and /stats reads live collection count.
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── path fix: allow `uvicorn src.api:app` from the project root ───────────────
ROOT = Path(__file__).parent.parent       # project root
SRC  = Path(__file__).parent             # src/
sys.path.insert(0, str(SRC))

from embed    import embed_chunks, EMBEDDING_MODEL
from extract  import extract_pages
from chunk    import chunk_pages
from store    import (
    store_chunks, make_prefixed_text,
    get_collection, CHROMA_DIR, COLLECTION_NAME,
)
from retrieve import retrieve
from generate import answer_with_sources, GENERATION_MODEL

# ── pipeline constants (same values as every other module) ────────────────────
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class IndexResponse(BaseModel):
    files_indexed: int  = Field(..., description="Number of PDF files processed")
    chunks_created: int = Field(..., description="Number of chunks upserted into ChromaDB")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int    = Field(5, ge=1, le=20, description="Number of chunks to retrieve (1–20)")


class SourceItem(BaseModel):
    file:    str = Field(..., description="PDF filename")
    page:    int = Field(..., description="1-indexed page number")
    quarter: str = Field(..., description="Quarter label, e.g. 'Q2 FY25'")


class AskResponse(BaseModel):
    answer:  str              = Field(..., description="Generated answer text")
    sources: list[SourceItem] = Field(..., description="Deduplicated source citations")


class StatsResponse(BaseModel):
    collection_name:  str = Field(..., description="ChromaDB collection name")
    chunk_count:      int = Field(..., description="Total chunks currently in the collection")
    embedding_model:  str = Field(..., description="Model used for embeddings")
    generation_model: str = Field(..., description="Model used for answer generation")


# ── lifespan: create one shared Mistral client at startup ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create a single shared Mistral client when the server starts.
    Store it in app.state so every request handler can reuse it.
    """
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY not set. Add it to your .env file and restart the server."
        )
    try:
        from mistralai.client.sdk import Mistral
    except ImportError:
        raise ImportError(
            "mistralai package not installed. Run: pip install -r requirements.txt"
        )
    app.state.mistral = Mistral(api_key=api_key)
    yield
    # Nothing to tear down — the client has no persistent connections to close.


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cisco Earnings RAG API",
    description=(
        "Retrieval-Augmented Generation over Cisco's FY25 quarterly earnings reports. "
        "Index PDFs via POST /index, then ask questions via POST /ask."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── POST /index ───────────────────────────────────────────────────────────────

@app.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Index PDF files",
    description=(
        "Upload one or more Cisco earnings PDFs. "
        "Runs the full extract → chunk → embed → store pipeline and returns "
        "the number of files processed and chunks upserted. "
        "Safe to call multiple times — upsert is idempotent."
    ),
)
async def index_pdfs(
    files: list[UploadFile] = File(..., description="One or more PDF files to index"),
) -> IndexResponse:
    client = app.state.mistral

    # Validate that all uploads are PDFs
    for uf in files:
        if uf.content_type not in ("application/pdf", "application/octet-stream"):
            if not (uf.filename or "").lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"File '{uf.filename}' does not appear to be a PDF.",
                )

    all_pages: list[dict] = []

    # Save uploads to a temp dir, extract page-by-page
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for uf in files:
            dest = tmp_path / (uf.filename or "upload.pdf")
            dest.write_bytes(await uf.read())
            all_pages.extend(extract_pages(dest))

    # Chunk
    chunks = chunk_pages(all_pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Embed (quarter-prefixed, matching the store.py convention)
    prefixed_chunks = [{**c, "text": make_prefixed_text(c)} for c in chunks]
    vectors = embed_chunks(prefixed_chunks, client=client)

    # Store (upsert — safe to re-run)
    store_chunks(chunks, vectors, persist_dir=CHROMA_DIR)

    return IndexResponse(
        files_indexed=len(files),
        chunks_created=len(chunks),
    )


# ── POST /ask ─────────────────────────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question",
    description=(
        "Retrieve the top_k most relevant chunks for `question` and generate a "
        "grounded answer.  Every numerical figure in the answer includes its unit "
        "and time period.  If the context does not contain the answer, the model "
        "says so plainly rather than guessing."
    ),
)
async def ask(body: AskRequest) -> AskResponse:
    client = app.state.mistral

    # Check that the collection has data before attempting retrieval
    collection = get_collection(CHROMA_DIR)
    if collection.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "The index is empty. "
                "Call POST /index with your PDFs before asking questions."
            ),
        )

    # Retrieve relevant chunks
    chunks = retrieve(
        body.question,
        top_k=body.top_k,
        debug=False,
        mistral_client=client,
    )

    # Generate answer + extract sources
    result = answer_with_sources(body.question, chunks, client=client)

    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )


# ── GET /stats ────────────────────────────────────────────────────────────────

@app.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Index and model statistics",
    description=(
        "Returns a live snapshot of the collection state and the model "
        "constants used for embedding and generation."
    ),
)
async def stats() -> StatsResponse:
    collection = get_collection(CHROMA_DIR)
    return StatsResponse(
        collection_name=COLLECTION_NAME,
        chunk_count=collection.count(),
        embedding_model=EMBEDDING_MODEL,
        generation_model=GENERATION_MODEL,
    )
