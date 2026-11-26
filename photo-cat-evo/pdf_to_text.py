"""
Step 1: PDF (+ optional Supporting Information) -> one plain-text file.

Usage:
    python pdf_to_text.py papers/paper_001.pdf
    python pdf_to_text.py papers/paper_001.pdf papers/paper_001_SI.pdf
Output:
    text/paper_001.txt
"""
import os
import re
import sys

import pymupdf  # pip install pymupdf

from config import TEXT_DIR


def _page_text(page) -> str:
    """Text of one page, keeping only horizontal lines.
    This drops rotated text such as the publisher's vertical 'Downloaded from ...' watermark,
    which would otherwise get mixed into the sentences."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:            # 0 = text block (1 = image)
            continue
        lines = []
        for line in block["lines"]:
            dx, dy = line["dir"]
            if abs(dy) > 0.1:                 # rotated / vertical line -> skip
                continue
            lines.append("".join(span["text"] for span in line["spans"]))
        if lines:
            out.append("\n".join(lines))
    return "\n\n".join(out)


def pdf_to_text(path: str) -> str:
    """Extract the text of every page."""
    doc = pymupdf.open(path)
    pages = [_page_text(page) for page in doc]
    doc.close()
    return "\n".join(pages)


def remove_references(text: str) -> str:
    """Cut the reference list (long and useless for extraction).
    Only cuts if the 'References' heading is in the second half of the paper,
    so a mention of 'references' early in the text is not mistaken for the heading."""
    matches = list(re.finditer(r"^\s*(references|reference list|bibliography)\s*$",
                               text, flags=re.IGNORECASE | re.MULTILINE))
    if matches and matches[-1].start() > len(text) * 0.5:
        return text[: matches[-1].start()]
    return text


def clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)        # join words split by hyphenation
    text = re.sub(r"[ \t]+", " ", text)             # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse empty lines
    return text.strip()


def convert(main_pdf: str, si_pdf: str | None = None) -> str:
    text = clean(remove_references(pdf_to_text(main_pdf)))
    if si_pdf:
        si = clean(pdf_to_text(si_pdf))
        text += "\n\n===== SUPPORTING INFORMATION =====\n\n" + si
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main_pdf = sys.argv[1]
    si_pdf = sys.argv[2] if len(sys.argv) > 2 else None

    text = convert(main_pdf, si_pdf)
    os.makedirs(TEXT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(main_pdf))[0]
    out = os.path.join(TEXT_DIR, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {out}  ({len(text):,} characters, ~{len(text)//4:,} tokens)")
