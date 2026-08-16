"""
src/restart_test.py — Stage 5: Persistence verification.

Purpose
-------
Prove that ChromaDB actually persists data across process restarts.
This script does NOT re-index anything.  It only reads the collection.

Two-pass test
-------------
Pass 1 — Connect, read count, disconnect (client goes out of scope).
Pass 2 — Reconnect fresh, read count again.
Report whether the two counts match.

Real restart test
-----------------
Run this script once, close your terminal, open a new one, activate
the venv, and run it again.  The count from a fresh process should
match what was indexed in Stage 5.

Usage
-----
    source venv/bin/activate
    python src/restart_test.py
"""

import sys
from pathlib import Path

# Allow "python src/restart_test.py" from project root
sys.path.insert(0, str(Path(__file__).parent))

from store import COLLECTION_NAME, CHROMA_DIR, get_collection


def read_count() -> int:
    """Open a fresh ChromaDB connection, read count, let the client close."""
    collection = get_collection(CHROMA_DIR)
    count = collection.count()
    # client goes out of scope here — no explicit close needed with PersistentClient
    return count


def main() -> None:
    sep = "─" * 56

    print(f"\n{sep}")
    print("  ChromaDB Persistence Test")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Store path : {CHROMA_DIR.resolve()}")
    print(sep)

    if not CHROMA_DIR.exists():
        print("\n  ⚠️  chroma_db/ directory not found.")
        print("  Run Stage 5 indexing first:")
        print("      source venv/bin/activate")
        print("      python src/store.py")
        print()
        sys.exit(1)

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    print("\n  Pass 1 — opening collection …")
    count_1 = read_count()
    print(f"  Chunk count (pass 1) : {count_1}")

    if count_1 == 0:
        print("\n  ⚠️  Collection exists but is empty.")
        print("  Run Stage 5 indexing first:")
        print("      python src/store.py")
        print()
        sys.exit(1)

    # ── Pass 2 ───────────────────────────────────────────────────────────────
    print("\n  Pass 2 — reconnecting (simulates process restart within same run) …")
    count_2 = read_count()
    print(f"  Chunk count (pass 2) : {count_2}")

    # ── Result ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    if count_1 == count_2:
        print(f"  ✅ PASS — counts match ({count_1} == {count_2})")
        print("     Data survives reconnection within the same process.")
    else:
        print(f"  ❌ FAIL — counts differ ({count_1} vs {count_2})")
        print("     Something is wrong with persistence — investigate.")
    print(f"{sep}\n")

    print("  ── Real restart test ──────────────────────────────────────")
    print("  To prove persistence across a true process boundary:")
    print()
    print("  1. Note the count above.")
    print("  2. Close this terminal completely (or open a new one).")
    print("  3. In the new terminal, from the project root:")
    print()
    print("       source venv/bin/activate")
    print("       python src/restart_test.py")
    print()
    print("  4. The count should be identical — ChromaDB wrote to disk.")
    print(f"  {'─' * 54}\n")


if __name__ == "__main__":
    main()
