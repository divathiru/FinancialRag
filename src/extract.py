"""
src/extract.py — Stage 2: Page-by-page text extraction from Cisco PDFs.

Public API
----------
extract_pages(pdf_path) -> list[dict]
    Returns one dict per page:
      {
        "file": str,   # bare filename, e.g. "Cisco_Q1_FY26.pdf"
        "page": int,   # 1-indexed page number
        "text": str,   # raw extracted text (no cleanup)
      }

extract_all(data_dir) -> list[dict]
    Runs extract_pages() over every *.pdf in data_dir and
    returns one flat list sorted by (file, page).

Usage (standalone)
------------------
    python src/extract.py
"""

import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data"

# ── core extraction ──────────────────────────────────────────────────────────

def extract_pages(pdf_path: Path | str) -> list[dict]:
    """
    Open a single PDF and return a list of page dicts.

    Each dict carries:
      "file"  — bare filename (not the full path), used for citations later.
      "page"  — 1-indexed page number.
      "text"  — raw text as returned by PyMuPDF (no stripping, no cleanup).

    No footer/header removal is done at this stage; that is acceptable noise.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for idx in range(doc.page_count):
        text = doc[idx].get_text("text")   # plain-text extraction
        pages.append(
            {
                "file": pdf_path.name,     # bare filename for citation
                "page": idx + 1,           # 1-indexed
                "text": text,
            }
        )
    doc.close()
    return pages


def extract_all(data_dir: Path | str = DATA_DIR) -> list[dict]:
    """
    Run extract_pages() over all *.pdf files in data_dir.
    Returns a flat list sorted by (file, page).
    """
    data_dir = Path(data_dir)
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}")

    all_pages: list[dict] = []
    for pdf_path in pdf_files:
        all_pages.extend(extract_pages(pdf_path))

    return all_pages   # already sorted because pdf_files is sorted and pages are in order


# ── CLI summary ──────────────────────────────────────────────────────────────

def _print_separator(width: int = 72) -> None:
    print("─" * width)


def main() -> None:
    print(f"\nExtracting text from PDFs in: {DATA_DIR.resolve()}\n")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found. Place Cisco quarterly PDFs in data/ and re-run.")
        sys.exit(1)

    grand_total = 0
    all_pages: list[dict] = []

    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        all_pages.extend(pages)
        page_count = len(pages)
        grand_total += page_count

        # First 300 chars of page 1 (strip leading whitespace for readability)
        preview = pages[0]["text"].strip()[:300] if pages else "[NO TEXT]"

        _print_separator()
        print(f"File      : {pdf_path.name}")
        print(f"Pages     : {page_count}")
        print(f"\nPage 1 preview (first 300 chars):")
        for line in preview.splitlines():
            print(f"  {line}")
        print()

    _print_separator()
    print(f"\nTotal pages processed across all files : {grand_total}")
    print(f"Total files processed                  : {len(pdf_files)}")
    print(f"\nExtraction complete. Each page dict carries: file, page, text.\n")


if __name__ == "__main__":
    main()
