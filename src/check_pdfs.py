"""
src/check_pdfs.py — Stage 1: PDF selection / extractability check.

For each PDF in data/, prints:
  - page count
  - how many pages have extractable text (len > MIN_CHARS threshold)
  - first 300 characters of page 1 text
  - a clear WARNING if any page returns empty/thin text

Usage:
    python src/check_pdfs.py
"""

import sys
from pathlib import Path

# ── threshold: fewer than this many non-whitespace chars → treat page as empty
MIN_CHARS = 50

DATA_DIR = Path(__file__).parent.parent / "data"

# ── try to import PyMuPDF (installed as 'pymupdf', imported as 'fitz') ─────
try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def check_pdf(pdf_path: Path) -> dict:
    """Open a PDF and return extraction stats."""
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    empty_pages = []       # 0-indexed page numbers that fail the threshold
    page1_preview = ""

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")          # plain-text extraction
        stripped = text.strip()

        if page_num == 0:
            page1_preview = stripped[:300]    # first 300 chars of page 1

        if len(stripped) < MIN_CHARS:
            empty_pages.append(page_num + 1)  # 1-indexed for display

    doc.close()

    return {
        "filename": pdf_path.name,
        "total_pages": total_pages,
        "good_pages": total_pages - len(empty_pages),
        "empty_pages": empty_pages,           # list of 1-indexed page numbers
        "page1_preview": page1_preview,
    }


def print_separator(width: int = 80) -> None:
    print("─" * width)


def main() -> None:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR.resolve()}")
        print("Place your Cisco quarterly press-release PDFs there and re-run.")
        sys.exit(1)

    print(f"\nPDF Extraction Check  (threshold: >{MIN_CHARS} non-whitespace chars = 'extractable')")
    print(f"Source directory: {DATA_DIR.resolve()}\n")

    any_warnings = False

    for pdf_path in pdf_files:
        stats = check_pdf(pdf_path)
        has_empty = bool(stats["empty_pages"])
        if has_empty:
            any_warnings = True

        print_separator()
        status_tag = "⚠️  WARNING — empty pages detected" if has_empty else "✅ OK"
        print(f"File        : {stats['filename']}")
        print(f"Status      : {status_tag}")
        print(f"Pages       : {stats['total_pages']}  total  |  "
              f"{stats['good_pages']}  extractable  |  "
              f"{len(stats['empty_pages'])}  empty/thin")

        if has_empty:
            print(f"Empty pages : {stats['empty_pages']}")

        print(f"\nPage 1 preview (first 300 chars):")
        preview = stats["page1_preview"] if stats["page1_preview"] else "[NO TEXT EXTRACTED]"
        # indent the preview block for readability
        for line in preview.splitlines():
            print(f"  {line}")

        print()

    print_separator()

    if any_warnings:
        print("\n⚠️  One or more files contain pages with little or no extractable text.")
        print("   Those pages may be image-only scans — consider OCR pre-processing.\n")
    else:
        print("\n✅ All pages in all PDFs have extractable text. Ready for Stage 2.\n")


if __name__ == "__main__":
    main()
