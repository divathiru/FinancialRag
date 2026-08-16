"""
src/retrieve.py — Stage 6: Semantic retrieval from ChromaDB.

Public API
----------
retrieve(question, top_k=4, debug=False) -> list[dict]
    Embeds `question` using the SAME model as the indexed chunks
    (EMBEDDING_MODEL from embed.py), queries the ChromaDB collection,
    and returns the top_k most similar chunks.

    Each returned dict contains:
        {
            "id"       : str,          # stable chunk ID
            "text"     : str,          # stored (prefixed) chunk text
            "file"     : str,          # source filename
            "page"     : int,          # 1-indexed page number
            "quarter"  : str,          # e.g. "Q1 FY25"
            "distance" : float,        # cosine distance (lower = more similar)
        }

    When debug=True, each chunk's file, page, quarter, and first 150 chars
    of text are printed before the list is returned.  Always enable this
    during development so you can sanity-check retrieval before trusting
    any generated answer.

embed_question(question, client=None) -> list[float]
    Embed a single string with the same model used for indexing.
    Exported so the app layer can call it directly if needed.

Usage (standalone — interactive test loop)
------------------------------------------
    source venv/bin/activate
    python src/retrieve.py

    Type any question at the prompt; retrieved chunks are printed with
    full debug output.  Ctrl-C or empty line to exit.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Allow "python src/retrieve.py" from project root
sys.path.insert(0, str(Path(__file__).parent))

from embed import EMBEDDING_MODEL
from store import CHROMA_DIR, get_collection


# ── embedding ────────────────────────────────────────────────────────────────

def embed_question(question: str, client=None) -> list[float]:
    """
    Embed a single question string using the same model as the indexed chunks.

    Parameters
    ----------
    question : str
        The user's natural-language query.
    client : mistralai.client.sdk.Mistral | None
        Reuse an existing client, or create a fresh one from MISTRAL_API_KEY.

    Returns
    -------
    list[float]
        A 1024-dimensional vector (mistral-embed).

    Notes
    -----
    MUST use EMBEDDING_MODEL — the same constant used during indexing.
    Using a different model would make cosine similarity meaningless.
    """
    try:
        from mistralai.client.sdk import Mistral
    except ImportError:
        raise ImportError("mistralai not installed. Run: pip install -r requirements.txt")

    if client is None:
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY not set. Add it to your .env file.")
        client = Mistral(api_key=api_key)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=[question],
    )
    return response.data[0].embedding


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    top_k: int = 4,
    debug: bool = False,
    mistral_client=None,
) -> list[dict]:
    """
    Embed `question` and return the top_k most similar chunks from ChromaDB.

    Parameters
    ----------
    question : str
        The user's natural-language query.
    top_k : int
        Number of results to return.  Default 4.
    debug : bool
        When True, prints a formatted summary of every retrieved chunk
        (file, page, quarter, first 150 chars) before returning.
        Always use this during development to validate retrieval quality
        before trusting any generated answer.
    mistral_client : Mistral | None
        Reuse an existing Mistral client.  A fresh one is created if None.

    Returns
    -------
    list[dict]
        Sorted by ascending cosine distance (closest first).
        Each dict:
            "id"       — stable chunk ID
            "text"     — stored (quarter-prefixed) chunk text
            "file"     — source filename
            "page"     — 1-indexed page number
            "quarter"  — e.g. "Q1 FY25"
            "distance" — cosine distance (0 = identical, 2 = opposite)
    """
    # 1. Embed the question with the same model used for indexing
    question_vector = embed_question(question, client=mistral_client)

    # 2. Query ChromaDB
    collection = get_collection(CHROMA_DIR)
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # 3. Unpack into a clean list of dicts
    chunks = []
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
        chunks.append(
            {
                "id":       cid,
                "text":     doc,
                "file":     meta.get("file", ""),
                "page":     meta.get("page", 0),
                "quarter":  meta.get("quarter", "Unknown"),
                "distance": round(dist, 6),
            }
        )

    # 4. Debug output — always read this before looking at a generated answer
    if debug:
        _print_debug(question, chunks)

    return chunks


# ── debug printer ─────────────────────────────────────────────────────────────

def _print_debug(question: str, chunks: list[dict]) -> None:
    """Print a human-readable summary of retrieved chunks."""
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  Query  : {question}")
    print(f"  Chunks : {len(chunks)} retrieved")
    print(sep)

    for i, chunk in enumerate(chunks, 1):
        preview = chunk["text"].replace("\n", " ").strip()[:150]
        print(f"\n  [{i}] {chunk['quarter']}  |  {chunk['file']}  p.{chunk['page']}"
              f"  |  dist={chunk['distance']:.4f}")
        print(f"       {preview} …")

    print(f"\n{sep}\n")


# ── standalone interactive loop ───────────────────────────────────────────────

def main() -> None:
    sep = "═" * 64
    print(f"\n{sep}")
    print("  Stage 6 — Retrieval test (debug mode ON)")
    print(f"  Model      : {EMBEDDING_MODEL}")
    print(f"  Store path : {CHROMA_DIR.resolve()}")
    print(f"{sep}")
    print("  Type a question and press Enter to retrieve chunks.")
    print("  Press Ctrl-C or enter an empty line to exit.\n")

    while True:
        try:
            question = input("  Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if not question:
            print("  Exiting.")
            break

        try:
            chunks = retrieve(question, top_k=4, debug=True)
        except Exception as exc:
            print(f"\n  ERROR: {exc}\n")


if __name__ == "__main__":
    main()
