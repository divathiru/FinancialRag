"""
src/embed.py — Stage 4: Batch embedding with Mistral AI mistral-embed.

Public API
----------
EMBEDDING_MODEL : str
    Single source of truth for the model name used BOTH here (indexing) and
    for embedding user questions at query time.  Never hard-code the model
    name anywhere else — always import this constant.

embed_chunks(chunks, client=None, batch_size=50) -> list[list[float]]
    Embeds a list of chunk dicts in batches and returns a list of vectors,
    aligned 1-to-1 with the input list (chunks[i] ↔ vectors[i]).

    This function is NOT called on import.  It is an explicit indexing step
    that must be invoked deliberately (e.g. from src/ingest.py in Stage 5).

Usage (standalone — for verification only)
------------------------------------------
    python src/embed.py

    Extracts + chunks all PDFs, embeds the result, and prints:
      - total chunks embedded
      - embedding dimension
      - wall-clock time taken
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# ── model constant ───────────────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH.
# Import this from here everywhere embeddings are created — both during
# indexing (ingest.py) and during query time (retrieve.py / app).
# mistral-embed produces 1024-dimensional vectors.
EMBEDDING_MODEL = "mistral-embed"

# Batch size: Mistral recommends batches of up to 50 inputs per request
# to stay comfortably within token limits.
DEFAULT_BATCH_SIZE = 50


def embed_chunks(
    chunks: list[dict],
    client=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """
    Embed a list of chunk dicts using Mistral AI's mistral-embed model.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have at least a "text" key (output of chunk.chunk_pages).
    client : mistralai.Mistral | None
        Pass an existing client to reuse; one is created from MISTRAL_API_KEY
        in the environment if None.
    batch_size : int
        Number of texts per API call.  Default 50.

    Returns
    -------
    list[list[float]]
        One embedding vector per input chunk, same order (chunks[i] ↔ vectors[i]).
        Dimension is 1024 for mistral-embed.

    Notes
    -----
    - NOT called on import.  Call explicitly from an indexing script.
    - EMBEDDING_MODEL is used here AND must be used when embedding user
      questions — using a different model would make cosine similarity
      meaningless.
    """
    try:
        from mistralai.client.sdk import Mistral
    except ImportError:
        raise ImportError("mistralai package not installed. Run: pip install -r requirements.txt")

    if client is None:
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY not set. Add it to your .env file.")
        client = Mistral(api_key=api_key)

    texts = [chunk["text"] for chunk in chunks]
    vectors: list[list[float]] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=batch,
        )
        # response.data is a list of EmbeddingObject; order matches input
        batch_vectors = [item.embedding for item in response.data]
        vectors.extend(batch_vectors)

        # Progress indicator for large corpora
        end = min(batch_start + batch_size, len(texts))
        print(f"  Embedded chunks {batch_start + 1}–{end} of {len(texts)} …")

    assert len(vectors) == len(chunks), (
        f"Mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
    )
    return vectors


# ── standalone verification ──────────────────────────────────────────────────

def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from extract import extract_all
    from chunk import chunk_pages

    CHUNK_SIZE    = 1200   # chosen in Stage 3
    CHUNK_OVERLAP = 150

    print(f"\nEmbedding model : {EMBEDDING_MODEL}  (used for BOTH indexing and queries)")
    print(f"Chunk size      : {CHUNK_SIZE} / overlap {CHUNK_OVERLAP}\n")

    print("Step 1 — Extracting pages …")
    pages = extract_all()
    print(f"  Pages loaded: {len(pages)}")

    print("\nStep 2 — Chunking …")
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  Chunks created: {len(chunks)}")

    print(f"\nStep 3 — Embedding (model: {EMBEDDING_MODEL}) …")
    t0 = time.time()
    vectors = embed_chunks(chunks)
    elapsed = time.time() - t0

    dim = len(vectors[0]) if vectors else 0

    print(f"\n{'─' * 60}")
    print(f"  Total chunks embedded : {len(vectors)}")
    print(f"  Embedding dimension   : {dim}")
    print(f"  Time taken            : {elapsed:.1f}s")
    print(f"  Model                 : {EMBEDDING_MODEL}")
    print(f"{'─' * 60}")
    print("\n✅ Embedding verification complete.\n")
    print("NOTE: This standalone run embeds but does NOT persist to ChromaDB.")
    print("      Persistence happens in Stage 5 (src/ingest.py).\n")


if __name__ == "__main__":
    main()
