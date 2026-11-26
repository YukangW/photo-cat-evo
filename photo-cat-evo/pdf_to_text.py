"""
Step 1: PDF (+ optional Supporting Information) -> one plain-text file.
Tables are detected and written as Markdown tables (| col | col |) where they appear in the paper.

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

# a real caption: "Table 1 Physical...", "Table 2. Rates", "Table S1: ..." (not "Table S3 and Figures")
TABLE_CAPTION = re.compile(r"^[ \t]*Table[ \t]+S?\d+[ \t]*[.:|]?[ \t]+[A-Z(]", re.MULTILINE)


#tables
def _cell_text(words, cell) -> str:
    """Text of one cell = the words whose CENTER lies inside the cell.
    Keeps decimal points ('52.1', not '52 1 .') and ignores text just outside the cell (e.g. the caption)."""
    if cell is None:
        return ""
    x0, y0, x1, y1 = cell
    inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    inside.sort(key=lambda w: (w[5], w[6], w[7]))     # block, line, word order
    return " ".join(w[4] for w in inside).replace("|", "/")   # '|' would break the Markdown table


def _table_to_markdown(page, table) -> str:
    words = page.get_text("words")                    # (x0, y0, x1, y1, text, block, line, word)
    rows = [[_cell_text(words, c) for c in row.cells] for row in table.rows]
    rows = [r for r in rows if any(r)]              # drop empty rows
    if len(rows) < 2:
        return ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _find_tables(page) -> list:
    """Detect tables. First the default (ruled lines); if nothing is found but the page has a
    'Table N' caption, retry for journal-style tables that only have horizontal lines."""
    found = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    if not found and re.search(r"^\s*Table\s+S?\d+", page.get_text(), re.IGNORECASE | re.MULTILINE):
        found = [t for t in page.find_tables(vertical_strategy="text", horizontal_strategy="lines").tables
                 if t.row_count >= 2 and t.col_count >= 2]
    return found


# pages
def _overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Fraction of rectangle a that lies inside rectangle b."""
    inter = pymupdf.Rect(a) & pymupdf.Rect(b)
    return 0.0 if inter.is_empty or a.get_area() == 0 else inter.get_area() / a.get_area()


def _page_text(page) -> tuple[str, int]:
    """Text of one page with tables as Markdown. Returns (text, number_of_tables).
    - drops rotated text (e.g. the vertical 'Downloaded from ...' watermark)
    - text blocks inside a table are replaced by the clean Markdown version of that table"""
    tables = []
    for t in _find_tables(page):
        md = _table_to_markdown(page, t)
        if md:
            tables.append({"rect": pymupdf.Rect(t.bbox), "md": md, "done": False})

    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:                       # 0 = text block (1 = image)
            continue
        rect = pymupdf.Rect(block["bbox"])
        inside = [t for t in tables if _overlap(rect, t["rect"]) > 0.5]
        if inside:                                       # block belongs to a table
            for t in inside:
                if not t["done"]:
                    out.append(t["md"])
                    t["done"] = True
            continue
        lines = []
        for line in block["lines"]:
            dx, dy = line["dir"]
            if abs(dy) > 0.1:                            # rotated / vertical line -> skip
                continue
            lines.append("".join(span["text"] for span in line["spans"]))
        if lines:
            out.append("\n".join(lines))

    out += [t["md"] for t in tables if not t["done"]]    # tables with no overlapping text block
    return "\n\n".join(out), len(tables)


def pdf_to_text(path: str) -> tuple[str, int]:
    """Extract the text of every page. Returns (text, total_number_of_tables)."""
    doc = pymupdf.open(path)
    pages, n_tables = [], 0
    for page in doc:
        txt, n = _page_text(page)
        pages.append(txt)
        n_tables += n
    doc.close()
    return "\n".join(pages), n_tables


# cleaning
def remove_references(text: str) -> str:
    """Cut the reference list (long and useless for extraction), only if the heading is in the
    second half of the paper."""
    matches = list(re.finditer(r"^\s*(references|reference list|bibliography)\s*$",
                               text, flags=re.IGNORECASE | re.MULTILINE))
    if matches and matches[-1].start() > len(text) * 0.5:
        return text[: matches[-1].start()]
    return text


def clean(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "-", text)   # broken minus signs / control chars -> "-"
    text = re.sub(r"-\n(?=[a-z])", "", text)        # join words split by hyphenation
    text = re.sub(r"[ \t]+", " ", text)             # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse empty lines
    text = re.sub(r"\u00ad\s*", "", text)                       # soft hyphen at a line break: rejoin the word
    for lig, rep in {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}.items():
        text = text.replace(lig, rep)                           # ligatures -> normal letters
    return text.strip()


def convert(main_pdf: str, si_pdf: str | None = None) -> tuple[str, int]:
    text, n = pdf_to_text(main_pdf)
    text = clean(remove_references(text))
    if si_pdf:
        si, n_si = pdf_to_text(si_pdf)
        text += "SUPPORTING INFORMATION" + clean(si)
        n += n_si
    return text, n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main_pdf = sys.argv[1]
    si_pdf = sys.argv[2] if len(sys.argv) > 2 else None

    text, n_tables = convert(main_pdf, si_pdf)
    os.makedirs(TEXT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(main_pdf))[0]
    out = os.path.join(TEXT_DIR, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {out}  ({len(text):,} characters, ~{len(text)//4:,} tokens, {n_tables} tables)")
    n_captions = len(TABLE_CAPTION.findall(text))
    if n_captions > n_tables:
        print(f"Note: found {n_captions} 'Table N' captions but only {n_tables} tables were detected. "
              f"Check those tables in the .txt file (they may be images or unusual layouts).")
