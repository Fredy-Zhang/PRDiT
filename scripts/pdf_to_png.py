#!/usr/bin/env python3
"""Convert PDF first page to PNG for README inline display."""
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Install: pip install pymupdf")
    exit(1)

def main():
    repo = Path(__file__).resolve().parent.parent
    pdf_path = repo / "assets" / "overview.pdf"
    out_path = repo / "assets" / "overview.png"

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        exit(1)

    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150, alpha=False)
    pix.save(str(out_path))
    doc.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
