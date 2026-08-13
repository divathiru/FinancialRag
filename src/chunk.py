"""
src/chunk.py — Stage 3: Recursive character-level text chunking.

Public API
----------
chunk_pages(pages, chunk_size, chunk_overlap) -> list[dict]

    Parameters
    ----------
    pages : list[dict]
        Output of extract.extract_all() — each dict has "file", "page", "text".
    chunk_size : int
        Maximum character length of a single chunk.
    chunk_overlap : int
        Number of characters of overlap between consecutive chunks.

    Returns
    -------
    list[dict], each entry:
      {
        "file"        : str,   # source filename — preserved for citations
        "page"        : int,   # 1-indexed page where this chunk STARTS
        "chunk_index" : int,   # 0-indexed position within the full corpus
        "text"        : str,   # chunk text
      }

    The splitter used is RecursiveCharacterTextSplitter, which tries to break
    on [paragraph, newline, space, character] in that order — much better than
    a naive fixed-width split for financial prose and tables.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Chunk a flat list of page dicts into smaller text units.

    Each output chunk preserves the source "file" and "page" of the page
    it originated from (specifically, the page where the chunk *starts*).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators are tried in order; falls back to the next if the
        # previous one can't create a chunk ≤ chunk_size.
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[dict] = []
    for page in pages:
        raw_chunks = splitter.split_text(page["text"])
        for piece in raw_chunks:
            chunks.append(
                {
                    "file": page["file"],   # source filename — citation anchor
                    "page": page["page"],   # page this chunk STARTED on
                    "chunk_index": len(chunks),  # global position in corpus
                    "text": piece,
                }
            )

    return chunks
