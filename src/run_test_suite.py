"""
src/run_test_suite.py — Stage 11: 10-question evaluation suite.

Calls retrieve() and answer_with_sources() directly (no HTTP server needed).
A single shared Mistral client is created once and reused for all 10 questions.

For every question the script prints:
  • The question text
  • The generated answer
  • A deduplicated source list (quarter · file · page)
  • A compact chunk table (rank · quarter · file · page · distance · 120-char preview)
    so any suspicious answer can be diagnosed immediately without re-running

An "ANSWERED / REFUSED" tag is printed for each question.
Question 10 is the trap — it MUST be REFUSED to pass.

Flags
-----
  --verbose   Print full chunk text instead of 120-char previews
  --json      Also write results to run_test_suite_results.json in the project root

Usage
-----
    source venv/bin/activate
    python src/run_test_suite.py
    python src/run_test_suite.py --verbose
    python src/run_test_suite.py --json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow "python src/run_test_suite.py" from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from retrieve import retrieve
from generate import answer_with_sources, GENERATION_MODEL
from embed    import EMBEDDING_MODEL

# ── refusal sentinel ──────────────────────────────────────────────────────────
# Must match the exact phrase in the system prompt in generate.py
REFUSAL_PHRASE = "The provided context does not contain enough information"

TOP_K = 5

# ── question bank ─────────────────────────────────────────────────────────────
# 10 questions mapped to Cisco FY25 Q1–Q3 press releases.
# Question 10 is the trap — deliberately unanswerable from our 3 PDFs.

QUESTIONS = [
    {
        "id":    1,
        "topic": "Revenue in the latest quarter",
        "text":  (
            "What was Cisco's total revenue for Q3 FY25 "
            "(the quarter ended April 26, 2025), "
            "and how did it break down between product and services?"
        ),
        "must_refuse": False,
    },
    {
        "id":    2,
        "topic": "Net profit compared across quarters",
        "text":  (
            "How did Cisco's GAAP net income compare "
            "across Q1, Q2, and Q3 FY25?"
        ),
        "must_refuse": False,
    },
    {
        "id":    3,
        "topic": "Year-on-year revenue comparison",
        "text":  (
            "How did Cisco's total revenue for Q3 FY25 "
            "compare to Q3 FY24 on a year-over-year basis?"
        ),
        "must_refuse": False,
    },
    {
        "id":    4,
        "topic": "Management commentary on demand",
        "text":  (
            "What did Cisco's management say about customer "
            "demand trends and order momentum in the FY25 earnings releases?"
        ),
        "must_refuse": False,
    },
    {
        "id":    5,
        "topic": "Fastest-growing segment",
        "text":  (
            "Which Cisco product category or technology platform "
            "showed the fastest revenue growth in FY25 "
            "(across Q1, Q2, and Q3)?"
        ),
        "must_refuse": False,
    },
    {
        "id":    6,
        "topic": "Operating margin trend",
        "text":  (
            "How did Cisco's GAAP operating margin change "
            "from Q1 FY25 to Q2 FY25 to Q3 FY25?"
        ),
        "must_refuse": False,
    },
    {
        "id":    7,
        "topic": "Dividend declared",
        "text":  (
            "What quarterly cash dividend per share did Cisco declare "
            "in Q3 FY25, and how did it compare to the dividend "
            "declared in Q3 FY24?"
        ),
        "must_refuse": False,
    },
    {
        "id":    8,
        "topic": "Risks and headwinds",
        "text":  (
            "What business risks, macro headwinds, or forward-looking "
            "cautions did Cisco highlight in its FY25 quarterly "
            "earnings press releases?"
        ),
        "must_refuse": False,
    },
    {
        "id":    9,
        "topic": "Three-line summary",
        "text":  (
            "Give a three-sentence executive summary of Cisco's overall "
            "financial performance across Q1, Q2, and Q3 FY25, "
            "covering revenue, profit, and any notable strategic highlights."
        ),
        "must_refuse": False,
    },
    {
        "id":    10,
        "topic": "Trap — must be refused",
        "text":  (
            "What was the exact breakdown of Cisco's global headcount "
            "by country and business unit at the end of FY2020, "
            "and how did voluntary attrition rates differ by region?"
        ),
        "must_refuse": True,
    },
]


# ── formatting helpers ────────────────────────────────────────────────────────

WIDE = "═" * 76
SEP  = "─" * 76
THIN = "·" * 76


def _tag(answer: str, must_refuse: bool) -> str:
    """Return PASS/FAIL tag based on expected behaviour."""
    refused = REFUSAL_PHRASE.lower() in answer.lower()
    if must_refuse:
        return "✅ PASS — correctly REFUSED" if refused else "❌ FAIL — should have refused"
    else:
        return "✅ ANSWERED" if not refused else "⚠️  REFUSED (check chunks below)"


def _fmt_sources(sources: list[dict]) -> str:
    groups: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    for s in sources:
        key = (s["quarter"], s["file"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s["page"])
    parts = []
    for key in order:
        q, f = key
        pages = " · ".join(f"p.{p}" for p in groups[key])
        parts.append(f"{q}: {f} {pages}")
    return " | ".join(parts) if parts else "(none)"


def _print_chunk_table(chunks: list[dict], verbose: bool) -> None:
    """Print a compact table of retrieved chunks."""
    print(f"\n  Retrieved chunks (top_k={len(chunks)}):")
    print(f"  {'Rank':<5} {'Quarter':<10} {'File':<32} {'Pg':>3} {'Dist':>6}  Preview")
    print(f"  {'-'*4} {'-'*9} {'-'*31} {'--':>3} {'----':>6}  {'-'*40}")
    for i, c in enumerate(chunks, 1):
        preview = c["text"].replace("\n", " ").strip()
        if not verbose:
            preview = preview[:120] + ("…" if len(preview) > 120 else "")
        fname = c["file"][:30] + ("…" if len(c["file"]) > 30 else "")
        print(f"  [{i}]  {c['quarter']:<10} {fname:<32} {c['page']:>3} {c['distance']:>6.4f}  {preview}")
        if verbose and len(c["text"]) > 120:
            # Print full text indented
            for line in c["text"].splitlines():
                print(f"           {line}")
            print()


def _print_result(rec: dict, verbose: bool) -> None:
    q      = rec["question"]
    ans    = rec["answer"]
    srcs   = rec["sources"]
    chunks = rec["chunks"]
    tag    = rec["tag"]
    qid    = rec["id"]
    topic  = rec["topic"]
    elapsed = rec["elapsed"]

    print(f"\n{WIDE}")
    print(f"  [{qid}/10] {topic}")
    print(WIDE)
    print(f"\n  QUESTION:\n  {q}\n")
    print(SEP)
    print(f"\n  ANSWER ({elapsed:.1f}s)  [{tag}]:")
    for line in ans.splitlines():
        print(f"  {line}")
    print()
    print(SEP)
    print(f"\n  SOURCES: {_fmt_sources(srcs)}\n")
    _print_chunk_table(chunks, verbose=verbose)
    print()


# ── markdown row helper ───────────────────────────────────────────────────────

def _md_row(rec: dict) -> str:
    q   = rec["question"].replace("|", "\\|")
    ans = " ".join(rec["answer"].splitlines()).replace("|", "\\|")
    src = _fmt_sources(rec["sources"]).replace("|", "\\|")
    tag = rec["tag"].replace("|", "\\|")
    return f"| {rec['id']} | {q} | {ans} | {src} | {tag} |"


# ── main ──────────────────────────────────────────────────────────────────────

def main(verbose: bool = False, dump_json: bool = False) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not set. Add it to your .env file.")

    try:
        from mistralai.client.sdk import Mistral
    except ImportError:
        raise ImportError("mistralai not installed. Run: pip install -r requirements.txt")

    client = Mistral(api_key=api_key)

    print(f"\n{WIDE}")
    print("  Stage 11 — 10-Question Evaluation Suite")
    print(f"  Embedding model   : {EMBEDDING_MODEL}")
    print(f"  Generation model  : {GENERATION_MODEL}")
    print(f"  top_k             : {TOP_K}")
    print(f"  Total questions   : {len(QUESTIONS)}  (9 answerable, 1 trap)")
    print(f"  Verbose mode      : {verbose}")
    print(WIDE)

    records: list[dict] = []
    pass_count = 0

    for q_data in QUESTIONS:
        qid   = q_data["id"]
        print(f"\n  Running [{qid}/10] {q_data['topic']} …", end="", flush=True)

        t0     = time.time()
        chunks = retrieve(q_data["text"], top_k=TOP_K, debug=False, mistral_client=client)
        result = answer_with_sources(q_data["text"], chunks, client=client)
        elapsed = time.time() - t0

        tag = _tag(result["answer"], q_data["must_refuse"])
        if "PASS" in tag or "ANSWERED" in tag:
            pass_count += 1

        rec = {
            "id":      qid,
            "topic":   q_data["topic"],
            "question": q_data["text"],
            "answer":  result["answer"],
            "sources": result["sources"],
            "chunks":  chunks,
            "tag":     tag,
            "elapsed": elapsed,
        }
        records.append(rec)
        print(f"  done ({elapsed:.1f}s)  {tag}")

    # ── full detail printout ──────────────────────────────────────────────────
    print(f"\n\n{WIDE}")
    print("  FULL RESULTS (question by question)")
    print(WIDE)
    for rec in records:
        _print_result(rec, verbose=verbose)

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{WIDE}")
    print("  SUMMARY")
    print(WIDE)
    print(f"\n  Questions passed : {pass_count} / {len(QUESTIONS)}")
    print(f"  Questions failed : {len(QUESTIONS) - pass_count} / {len(QUESTIONS)}")
    print()

    # Markdown table (copy-paste ready)
    print(THIN)
    print("  MARKDOWN TABLE (copy-paste into README)")
    print(THIN)
    print("\n| # | Question | Answer | Sources | Status |")
    print("|---|----------|--------|---------|--------|")
    for rec in records:
        print(_md_row(rec))
    print()

    # ── optional JSON dump ────────────────────────────────────────────────────
    if dump_json:
        out_path = ROOT / "run_test_suite_results.json"
        # chunks contain float distances — fine for JSON serialisation
        json_records = [
            {k: v for k, v in r.items() if k != "chunks"}
            | {"chunks_summary": [
                {
                    "rank":     i + 1,
                    "quarter":  c["quarter"],
                    "file":     c["file"],
                    "page":     c["page"],
                    "distance": c["distance"],
                    "preview":  c["text"].replace("\n", " ")[:200],
                }
                for i, c in enumerate(r["chunks"])
            ]}
            for r in records
        ]
        with open(out_path, "w") as f:
            json.dump(json_records, f, indent=2)
        print(f"  JSON results written to: {out_path}")

    print(f"\n{WIDE}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 10-question evaluation suite over the Cisco Earnings RAG system."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full chunk text instead of 120-char previews.",
    )
    parser.add_argument(
        "--json", dest="dump_json", action="store_true",
        help="Write results to run_test_suite_results.json.",
    )
    args = parser.parse_args()
    main(verbose=args.verbose, dump_json=args.dump_json)
