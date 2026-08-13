"""
src/compare_chunk_sizes.py — Stage 3: Compare chunk_size=800 vs 1200.

Steps
-----
1. Extract all pages from data/ using src/extract.py
2. Chunk at 800/150 and at 1200/150
3. Print total chunk counts for each setting
4. Print 3 random chunks from each run (seeded for reproducibility)
5. Prompt you for a search term (e.g. "operating margin") and show the best
   matching chunk from each size, so you can see which captures a full table
   vs a fragment.

Usage
-----
    python src/compare_chunk_sizes.py
"""

import random
import sys
from pathlib import Path

# Allow "python src/compare_chunk_sizes.py" from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extract import extract_all
from chunk import chunk_pages

# ── constants ────────────────────────────────────────────────────────────────
CONFIGS = [
    {"chunk_size": 800,  "chunk_overlap": 150, "label": "800 / overlap 150"},
    {"chunk_size": 1200, "chunk_overlap": 150, "label": "1200 / overlap 150"},
]
RANDOM_SEED = 42          # fix seed so "random" samples are reproducible
SAMPLE_COUNT = 3          # how many random chunks to show per config


def _separator(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _print_chunk(chunk: dict, label: str) -> None:
    """Pretty-print a single chunk with its metadata."""
    print(f"  [{label}]")
    print(f"  Source : {chunk['file']}  page {chunk['page']}")
    print(f"  Length : {len(chunk['text'])} chars")
    print(f"  Text   :")
    for line in chunk["text"].splitlines():
        print(f"    {line}")
    print()


def find_best_chunk(chunks: list[dict], query: str) -> dict | None:
    """
    Score chunks by how many query words they contain (case-insensitive).
    Returns the highest-scoring chunk, or None if no match at all.
    """
    query_words = query.lower().split()
    best_chunk = None
    best_score = 0

    for chunk in chunks:
        text_lower = chunk["text"].lower()
        score = sum(1 for word in query_words if word in text_lower)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk if best_score > 0 else None


def main() -> None:
    # ── 1. Extract ───────────────────────────────────────────────────────────
    print("\nLoading pages …")
    try:
        pages = extract_all()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"Total pages loaded: {len(pages)}\n")

    # ── 2 & 3. Chunk + count ─────────────────────────────────────────────────
    results: dict[str, list[dict]] = {}
    _separator("═")
    print("CHUNK COUNT COMPARISON")
    _separator("═")
    for cfg in CONFIGS:
        chunks = chunk_pages(pages, cfg["chunk_size"], cfg["chunk_overlap"])
        results[cfg["label"]] = chunks
        print(f"  chunk_size={cfg['chunk_size']:>4}  overlap={cfg['chunk_overlap']}  →  {len(chunks):>4} chunks")
    print()

    # ── 4. Random samples ────────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)

    for cfg in CONFIGS:
        label = cfg["label"]
        chunks = results[label]
        sample = rng.sample(chunks, min(SAMPLE_COUNT, len(chunks)))

        _separator("─")
        print(f"  RANDOM SAMPLES  —  chunk_size={cfg['chunk_size']}  ({len(chunks)} total chunks)")
        _separator("─")
        for i, chunk in enumerate(sample, 1):
            print(f"\n  ── Sample {i} of {SAMPLE_COUNT} ──")
            _print_chunk(chunk, label)

    # ── 5. Interactive table-capture test ────────────────────────────────────
    _separator("═")
    print("FINANCIAL TABLE CAPTURE TEST")
    _separator("═")
    print(
        "\nEnter a search term to find the best matching chunk in each size.\n"
        "Try something like: operating margin\n"
        "                or: revenue  net income  total  gross margin\n"
    )

    try:
        query = input("Search term(s): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[Skipped — no search term provided]")
        return

    if not query:
        print("[No query entered — skipping table capture test]")
        return

    print()
    for cfg in CONFIGS:
        label = cfg["label"]
        chunks = results[label]
        best = find_best_chunk(chunks, query)

        _separator("─")
        print(f"  chunk_size={cfg['chunk_size']}  — best match for: '{query}'")
        _separator("─")
        if best:
            _print_chunk(best, label)
        else:
            print(f"  [No chunk contained any word from '{query}']\n")

    _separator("═")
    print("\nDone. Use the output above to pick your chunk size for Stage 4.\n")


if __name__ == "__main__":
    main()
