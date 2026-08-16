"""
src/generate.py — Stage 7: Answer generation via Mistral chat.

Public API
----------
GENERATION_MODEL : str
    The Mistral chat model used for answer generation.  Single source of
    truth — import this constant wherever you need to refer to the model.

build_context(retrieved_chunks) -> str
    Formats the retrieved chunks into a numbered, labeled context block
    for the prompt.  Exported so the app layer can inspect the exact
    context that was passed to the model.

answer(question, retrieved_chunks, client=None, debug=False) -> str
    The main generation function.  Builds a four-part prompt, calls Mistral
    at temperature=0.2, and returns the model's answer string.

    Rules enforced by the system prompt:
      1. Answer ONLY from the provided context — never from general knowledge.
      2. If the context doesn't contain the answer, say so plainly.
      3. Every figure must include its unit and time period explicitly
         (e.g. "$13.6 billion for Q2 FY25", never a bare number).

    When debug=True, the full context block + question are printed before
    the API call so you can verify what the model actually saw.

Usage (standalone — end-to-end test on a real question)
--------------------------------------------------------
    source venv/bin/activate
    python src/generate.py

    Retrieves chunks for a hard-coded test question, prints them, calls
    the model, and prints the answer side-by-side with the source chunks
    so every fact can be traced.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Allow "python src/generate.py" from the project root
sys.path.insert(0, str(Path(__file__).parent))

from retrieve import retrieve

# ── model constant ───────────────────────────────────────────────────────────
# Change this one line to swap the generation model globally.
GENERATION_MODEL = "mistral-small-latest"

# ── system prompt ────────────────────────────────────────────────────────────
# Written once here; never duplicated in app.py or any other caller.
_SYSTEM_PROMPT = """\
You are a precise financial analyst assistant specialising in Cisco's \
quarterly earnings reports.

RULES — follow every rule without exception:
1. Answer ONLY using the SOURCE PASSAGES provided below. \
   Never draw on any general knowledge about Cisco or the technology industry.
2. If the provided passages do not contain enough information to answer the \
   question, say clearly: "The provided context does not contain enough \
   information to answer this question." Do not guess or infer.
3. Every numerical figure you state MUST include:
   • its unit  (e.g. "billion dollars", "%", "million shares")
   • its time period  (e.g. "for Q2 FY25", "in the three months ended \
January 25 2025")
   Example of correct format: "$13.8 billion for Q1 FY25"
   Example of WRONG format  : "$13.8 billion" or "13.8"
4. When comparing quarters, label every number with its quarter explicitly.
5. Keep your answer concise and factual. Do not add commentary, opinion, \
   or information not present in the passages.\
"""


# ── context builder ───────────────────────────────────────────────────────────

def build_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered, labeled context block.

    Each chunk is introduced with its source metadata (quarter, file, page)
    so the model (and the human reviewer) can trace every fact back to its
    original passage.

    Parameters
    ----------
    retrieved_chunks : list[dict]
        Output of retrieve.retrieve() — each dict has text, file, page,
        quarter, distance.

    Returns
    -------
    str
        A formatted multi-line string ready to be inserted into the prompt.
    """
    lines = ["SOURCE PASSAGES:", ""]
    for i, chunk in enumerate(retrieved_chunks, 1):
        lines.append(
            f"[{i}] {chunk['quarter']}  |  {chunk['file']}  page {chunk['page']}"
        )
        lines.append(chunk["text"])
        lines.append("")   # blank line between passages
    return "\n".join(lines)


# ── generation ────────────────────────────────────────────────────────────────

def answer(
    question: str,
    retrieved_chunks: list[dict],
    client=None,
    debug: bool = False,
) -> str:
    """
    Build a four-part grounded prompt and call Mistral to generate an answer.

    The four parts are:
      1. System instruction  — role + strict grounding rules
      2. Source passages     — the chunks from retrieve(), each labeled
      3. User question       — the original question
      4. Answer instruction  — explicit reminder to cite period and unit

    Parameters
    ----------
    question : str
        The user's natural-language question.
    retrieved_chunks : list[dict]
        Output of retrieve.retrieve().  The model sees ONLY these chunks —
        nothing is added, nothing is dropped.
    client : mistralai.client.sdk.Mistral | None
        Reuse an existing Mistral client, or create a fresh one.
    debug : bool
        When True, prints the full context block and question before the
        API call so you can verify exactly what the model received.

    Returns
    -------
    str
        The model's answer text, stripped of leading/trailing whitespace.
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

    # ── Build the user message ───────────────────────────────────────────────
    context_block = build_context(retrieved_chunks)

    user_message = (
        f"{context_block}\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (include unit and time period for every figure):"
    )

    if debug:
        _print_debug_prompt(user_message)

    # ── Call the model ───────────────────────────────────────────────────────
    response = client.chat.complete(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # low temperature = more deterministic, fewer hallucinations
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()


# ── debug helpers ─────────────────────────────────────────────────────────────

def _print_debug_prompt(user_message: str) -> None:
    """Print the exact user message sent to the model."""
    sep = "─" * 64
    print(f"\n{sep}")
    print("  DEBUG — exact prompt sent to model")
    print(sep)
    print(user_message)
    print(sep + "\n")


def _print_side_by_side(question: str, chunks: list[dict], generated: str) -> None:
    """
    Print retrieved chunks alongside the generated answer so every fact
    in the answer can be traced to a source passage.
    """
    wide = "═" * 64
    sep  = "─" * 64

    print(f"\n{wide}")
    print("  QUESTION")
    print(wide)
    print(f"  {question}")

    print(f"\n{wide}")
    print("  SOURCE CHUNKS (what the model was given)")
    print(wide)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  [{i}] {chunk['quarter']}  |  {chunk['file']}  p.{chunk['page']}"
              f"  |  dist={chunk['distance']:.4f}")
        # Print full chunk text, indented
        for line in chunk["text"].splitlines():
            print(f"      {line}")

    print(f"\n{wide}")
    print("  GENERATED ANSWER")
    print(wide)
    for line in generated.splitlines():
        print(f"  {line}")
    print(f"\n{wide}\n")


# ── standalone end-to-end test ────────────────────────────────────────────────

def main() -> None:
    TEST_QUESTION = "What was Cisco's total revenue for Q2 FY25, and how did it compare to Q2 FY24?"

    print("\n" + "═" * 64)
    print("  Stage 7 — Generation test (end-to-end)")
    print(f"  Chat model : {GENERATION_MODEL}")
    print("═" * 64)
    print(f"\n  Test question:\n  {TEST_QUESTION}\n")

    # Step 1: retrieve
    print("  Step 1 — Retrieving chunks (debug=True) …")
    chunks = retrieve(TEST_QUESTION, top_k=5, debug=True)

    # Step 2: generate
    print("  Step 2 — Generating answer …")
    generated = answer(TEST_QUESTION, chunks, debug=False)

    # Step 3: side-by-side display for fact-checking
    _print_side_by_side(TEST_QUESTION, chunks, generated)


if __name__ == "__main__":
    main()
