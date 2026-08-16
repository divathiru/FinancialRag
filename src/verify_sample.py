"""
src/verify_sample.py — Stage 8: Manual verification of answer quality + sources.

Runs four questions through the full RAG pipeline (retrieve → generate):
  • Q1–Q3: factual questions whose answers are known to exist in the PDFs
  • Q4:    an out-of-scope question that must trigger a clean refusal

Output is formatted so each block can be copy-pasted straight into a markdown
table in the README.

Usage
-----
    source venv/bin/activate
    python src/verify_sample.py
"""

import os
import sys
from pathlib import Path

# Allow "python src/verify_sample.py" from project root
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from retrieve import retrieve
from generate import answer_with_sources, GENERATION_MODEL

# ── question bank ─────────────────────────────────────────────────────────────

# Three hand-picked questions where the answer is present in our 3 PDFs.
# One clearly out-of-scope question that must produce a refusal.
QUESTIONS = [
    {
        "label": "Q1 (in-scope)",
        "text":  "What was Cisco's total revenue for Q1 FY25?",
    },
    {
        "label": "Q2 (in-scope)",
        "text":  "What was Cisco's GAAP net income for Q3 FY25?",
    },
    {
        "label": "Q3 (in-scope)",
        "text":  "How many shares did Cisco repurchase during Q2 FY25, and at what total cost?",
    },
    {
        "label": "Q4 (out-of-scope — must refusal)",
        "text":  (
            "What was Cisco's total research and development headcount "
            "broken down by country as of the end of FY2023?"
        ),
    },
]

TOP_K = 5   # chunks retrieved per question


# ── formatting helpers ────────────────────────────────────────────────────────

def _fmt_sources(sources: list[dict]) -> str:
    """
    Render the source list as a compact, markdown-safe string.

    Example output (single cell):
        Q1 FY25: Q1FY25-Press-Release.pdf p.3 · p.5
    Sources that share the same file are grouped and their pages joined with ·.
    """
    # Group pages by (quarter, file), preserving first-appearance order
    groups: dict[tuple, list[int]] = {}
    order:  list[tuple]            = []
    for s in sources:
        key = (s["quarter"], s["file"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s["page"])

    parts = []
    for key in order:
        quarter, fname = key
        pages = " · ".join(f"p.{p}" for p in groups[key])
        parts.append(f"{quarter}: {fname} {pages}")

    return " | ".join(parts)


def _print_block(label: str, question: str, result: dict, idx: int) -> None:
    """
    Print one question's output in two formats:
      1. Human-readable block (for quick eyeballing in the terminal)
      2. Markdown table row (copy-paste ready)
    """
    answer_text = result["answer"]
    sources_str = _fmt_sources(result["sources"])

    wide = "═" * 72
    sep  = "─" * 72

    # ── human-readable block ─────────────────────────────────────────────────
    print(f"\n{wide}")
    print(f"  [{idx}] {label}")
    print(wide)
    print(f"  Question : {question}")
    print(sep)
    print(f"  Answer   :")
    for line in answer_text.splitlines():
        print(f"             {line}")
    print(sep)
    print(f"  Sources  : {sources_str}")
    print(wide)

    # ── markdown table row ───────────────────────────────────────────────────
    # Newlines in the answer would break the table cell; collapse to a space.
    answer_flat = " ".join(answer_text.splitlines())
    print(f"\n  MARKDOWN ROW (copy-paste):")
    print(f"  | {question} | {answer_flat} | {sources_str} |")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not set. Add it to your .env file.")

    # Build a single shared Mistral client to avoid creating one per call
    try:
        from mistralai.client.sdk import Mistral
    except ImportError:
        raise ImportError("mistralai not installed. Run: pip install -r requirements.txt")

    client = Mistral(api_key=api_key)

    wide = "═" * 72
    print(f"\n{wide}")
    print("  Stage 8 — Verify Sample  (answer_with_sources)")
    print(f"  Generation model : {GENERATION_MODEL}")
    print(f"  Questions        : {len(QUESTIONS)}  ({len(QUESTIONS)-1} in-scope, 1 out-of-scope)")
    print(f"  top_k            : {TOP_K}")
    print(wide)

    for idx, q in enumerate(QUESTIONS, 1):
        label    = q["label"]
        question = q["text"]

        print(f"\n  Running [{idx}/{len(QUESTIONS)}] {label} …")

        # Retrieve
        chunks = retrieve(question, top_k=TOP_K, debug=False, mistral_client=client)

        # Generate + extract sources
        result = answer_with_sources(question, chunks, client=client, debug=False)

        # Display
        _print_block(label, question, result, idx)

    # ── summary table header (for README) ────────────────────────────────────
    print("\n\n" + "─" * 72)
    print("  FULL MARKDOWN TABLE  (paste into README)")
    print("─" * 72)
    print("\n| # | Question | Answer | Sources |")
    print("|---|----------|--------|---------|")
    print("  (see individual MARKDOWN ROW lines above)")
    print("─" * 72 + "\n")


if __name__ == "__main__":
    main()
