"""
src/store.py — Stage 5: Persist chunks + embeddings into ChromaDB.

Public API
----------
COLLECTION_NAME : str
    Name of the ChromaDB collection. Single source of truth — import this
    constant wherever you need to open the same collection.

CHROMA_DIR : Path
    Default on-disk path for the persistent store (./chroma_db).
    Override via the persist_dir parameter if needed.

derive_quarter(filename) -> str
    Extracts a human-readable quarter label from a PDF filename.
    Examples:
      "Q1FY25-Press-Release.pdf"  ->  "Q1 FY25"
      "Cisco_Q2_FY26.pdf"         ->  "Q2 FY26"
    Returns "Unknown" if no recognisable pattern is found.

make_chunk_id(filename, chunk_index) -> str
    Builds a stable, deterministic ID for a chunk so that re-running
    indexing upserts (overwrites) existing rows instead of duplicating.
    Format: "<bare_filename>__chunk_<zero-padded-index>"

get_collection(persist_dir) -> chromadb.Collection
    Opens (or creates) the persistent ChromaDB collection.
    Call this whenever you need a handle to the store — at query time
    as well as at index time.

store_chunks(chunks, vectors, persist_dir) -> int
    The main indexing function.  Upserts all chunk embeddings + metadata
    into ChromaDB in one batched call.  Returns the total count of items
    in the collection after the upsert.

Usage (standalone — full pipeline run)
---------------------------------------
    python src/store.py

    Runs the full pipeline (extract → chunk → embed → store) and prints
    a summary.  Safe to re-run; existing rows are overwritten, not duplicated.
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── constants ────────────────────────────────────────────────────────────────

COLLECTION_NAME = "cisco_financials"

# Default location for the persistent store — relative to project root.
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# ── helpers ──────────────────────────────────────────────────────────────────

def derive_quarter(filename: str) -> str:
    """
    Extract a human-readable quarter label from a PDF filename.

    Handles two filename conventions seen in this project:
      • "Q1FY25-Press-Release.pdf"  ->  "Q1 FY25"
      • "Cisco_Q2_FY26.pdf"         ->  "Q2 FY26"

    Returns "Unknown" if neither pattern matches.
    """
    name = Path(filename).stem  # strip extension

    # Pattern A: Q1FY25  (no separator between Q and FY)
    m = re.search(r"(Q\d)(FY\d{2,4})", name, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} {m.group(2).upper()}"

    # Pattern B: Q1_FY25 or Q1-FY25 or Q1 FY25  (separator present)
    m = re.search(r"(Q\d)[_\-\s](FY\d{2,4})", name, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} {m.group(2).upper()}"

    return "Unknown"


def make_chunk_id(filename: str, chunk_index: int) -> str:
    """
    Build a stable, deterministic ID for a chunk.

    Using a deterministic ID (not a random UUID) means that re-running
    indexing calls collection.upsert(), which overwrites the existing row
    rather than inserting a duplicate.

    Format: "<bare_filename>__chunk_<zero-padded-5-digit-index>"
    Example: "Q1FY25-Press-Release.pdf__chunk_00042"
    """
    return f"{filename}__chunk_{chunk_index:05d}"


def get_collection(persist_dir: Path | str = CHROMA_DIR):
    """
    Open (or create) the ChromaDB persistent collection.

    Parameters
    ----------
    persist_dir : Path | str
        Directory where ChromaDB stores its files.  Created automatically
        if it does not exist.

    Returns
    -------
    chromadb.Collection
        A handle to the "cisco_financials" collection, ready for
        upsert() or query() calls.
    """
    try:
        import chromadb
    except ImportError:
        raise ImportError("chromadb not installed. Run: pip install -r requirements.txt")

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # cosine similarity is standard for embedding search
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def make_prefixed_text(chunk: dict) -> str:
    """
    Prefix a chunk's text with its quarter label.

    Example output: "[Cisco Q1 FY25] Revenue for the quarter was ..."

    Why this matters (the quarter-fix)
    -----------------------------------
    Without the prefix, the quarter is only in *metadata*, which is never
    seen by the embedding model.  That means a question like
    "What was Q2 revenue?" cannot distinguish between a Q1 and Q2 chunk
    on semantic grounds alone.

    By baking the label into the embedded text, the quarter becomes part
    of the semantic vector.  The same prefixed text is stored as the
    ChromaDB document, so the LLM receives the label in retrieved context.
    """
    quarter = derive_quarter(chunk["file"])
    return f"[Cisco {quarter}] {chunk['text']}"


def store_chunks(
    chunks: list[dict],
    vectors: list[list[float]],
    persist_dir: Path | str = CHROMA_DIR,
) -> int:
    """
    Upsert all chunk embeddings, texts, and metadata into ChromaDB.

    Parameters
    ----------
    chunks : list[dict]
        Output of chunk.chunk_pages() — each dict has "file", "page",
        "chunk_index", "text".
    vectors : list[list[float]]
        Aligned embedding vectors from embed_prefixed_chunks() (i.e. the
        vectors were computed on the quarter-prefixed text, not raw text).
        Must satisfy len(vectors) == len(chunks).
    persist_dir : Path | str
        Directory for the persistent ChromaDB store.  Default: ./chroma_db.

    Returns
    -------
    int
        Total number of items in the collection after the upsert.

    Notes
    -----
    - Uses upsert() so re-running is safe: existing chunks are overwritten,
      not duplicated.
    - Each chunk gets a stable ID from make_chunk_id(filename, chunk_index).
    - The stored document is the PREFIXED text ("[Cisco Q1 FY25] ..."),
      matching what was embedded.  Metadata: {file, page, quarter}.
    """
    if len(vectors) != len(chunks):
        raise ValueError(
            f"len(vectors)={len(vectors)} != len(chunks)={len(chunks)}"
        )

    collection = get_collection(persist_dir)

    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for chunk, vector in zip(chunks, vectors):
        quarter = derive_quarter(chunk["file"])
        ids.append(make_chunk_id(chunk["file"], chunk["chunk_index"]))
        embeddings.append(vector)
        # Store the prefixed text — matches what was embedded
        documents.append(make_prefixed_text(chunk))
        metadatas.append(
            {
                "file":    chunk["file"],
                "page":    chunk["page"],
                "quarter": quarter,
            }
        )

    # upsert = insert-or-overwrite (idempotent)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return collection.count()


# ── standalone pipeline run ──────────────────────────────────────────────────

def main() -> None:
    import time

    # Allow "python src/store.py" from the project root
    sys.path.insert(0, str(Path(__file__).parent))
    from extract import extract_all
    from chunk import chunk_pages
    from embed import embed_chunks, EMBEDDING_MODEL

    CHUNK_SIZE    = 1200
    CHUNK_OVERLAP = 150

    print("\n" + "═" * 60)
    print("  Stage 5 — Indexing pipeline")
    print("═" * 60)
    print(f"  Embedding model  : {EMBEDDING_MODEL}")
    print(f"  Collection       : {COLLECTION_NAME}")
    print(f"  Store path       : {CHROMA_DIR.resolve()}")
    print()

    # ── 1. Extract ───────────────────────────────────────────────────────────
    print("Step 1 — Extracting pages …")
    pages = extract_all()
    print(f"  Pages loaded: {len(pages)}")

    # ── 2. Chunk ─────────────────────────────────────────────────────────────
    print("\nStep 2 — Chunking …")
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  Chunks created: {len(chunks)}")

    # Quick sanity-check: show quarter labels for the unique filenames found
    unique_files = sorted({c["file"] for c in chunks})
    print("\n  Quarter labels derived:")
    for f in unique_files:
        print(f"    {f}  →  {derive_quarter(f)}")

    # ── 3. Embed ─────────────────────────────────────────────────────────────
    # Quarter-fix: embed the PREFIXED version of each chunk so that quarter
    # identity becomes part of the semantic vector, not just a metadata tag.
    print(f"\nStep 3 — Embedding ({len(chunks)} chunks via {EMBEDDING_MODEL}) …")
    print("  (using quarter-prefixed text, e.g. '[Cisco Q1 FY25] ...')")
    prefixed_chunks = [
        {**c, "text": make_prefixed_text(c)} for c in chunks
    ]
    t0 = time.time()
    vectors = embed_chunks(prefixed_chunks)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  (dim={len(vectors[0])})")

    # ── 4. Store ─────────────────────────────────────────────────────────────
    print(f"\nStep 4 — Upserting into ChromaDB …")
    total = store_chunks(chunks, vectors)

    print(f"\n{'═' * 60}")
    print(f"  ✅ Indexing complete")
    print(f"  Chunks upserted  : {len(chunks)}")
    print(f"  Collection total : {total}")
    print(f"  Store path       : {CHROMA_DIR.resolve()}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
