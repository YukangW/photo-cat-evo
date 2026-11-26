"""
Combine the extraction results of many papers into ONE table: one row per sample, one column per parameter.

Input:  a folder with results/<paper>_extraction.csv files (made by `uv run PhotoCatEvo ...`)
Output: one CSV with
    paper | doi | sample_name | is_main | <parameter> | <parameter>_unit | <next parameter> | ...

Values (standard units are used when the conversion worked, otherwise the paper's own value and unit):
    specific value      -> the value                       3.05        -> 3.05
    range "x-y"         -> the average                     2.6-2.8     -> 2.7
    bound ">x" / "<x"   -> x                               >160        -> 160
    text (e.g. "air")   -> kept as text
Rows that are not_reported or failed are skipped.

Usage:
    uv run PhotoCatEvo-table results/
    uv run PhotoCatEvo-table results/ -o results/all_samples.csv
    uv run PhotoCatEvo-table results/ --with-source      # also a <parameter>_source column
"""
import argparse
import csv
import glob
import os
import re
import unicodedata

from .units import _NUMBER

ID_COLUMNS = ["paper", "doi", "sample_name", "is_main"]
KEEP = ("verified", "range_only")


def _norm_name(s: str) -> str:
    """Sample names that differ only in case, spaces or dash style are the same sample."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[\s\-‐-―−_]+", "", s)


def range_to_number(text: str) -> tuple[float | None, str]:
    """'2.6-2.8' -> (2.7, 'range average'), '>160' -> (160, 'bound'), '' -> (None, '')."""
    t = unicodedata.normalize("NFKC", text or "").strip()
    if not t:
        return None, ""
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)                # 1,205 -> 1205
    numbers = [float(n) for n in _NUMBER.findall(t)]
    if not numbers:
        return None, ""
    if len(numbers) >= 2:
        lo, hi = numbers[0], numbers[1]
        return float(f"{(lo + hi) / 2:.10g}"), "range average"
    return numbers[0], "bound" if re.search(r"[<>≤≥]|less|more|greater|above|below|up to", t, re.I) else "range"


def _as_number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def cell_value(row: dict) -> tuple[object, str, str]:
    """(value, unit, source) for one extracted row. Standard units first, the paper's own as fallback."""
    std_unit = (row.get("standard_unit") or "").strip()
    unit = (row.get("unit") or "").strip()

    if (row.get("standard_value") or "").strip():                     # converted specific value
        n = _as_number(row["standard_value"])
        return (n if n is not None else row["standard_value"].strip()), std_unit, "value"
    if (row.get("value") or "").strip():                              # not converted (or text)
        v = row["value"].strip()
        n = _as_number(v)
        return (n if n is not None else v), unit, "value (paper unit)" if n is not None else "text"
    for col, u in (("standard_range", std_unit), ("range", unit)):    # range / bound
        n, how = range_to_number(row.get(col) or "")
        if n is not None:
            return n, u, how + ("" if col == "standard_range" else " (paper unit)")
    return None, "", ""


def organize(folder: str, out_csv: str | None = None, with_source: bool = False) -> str:
    files = sorted(glob.glob(os.path.join(folder, "*_extraction.csv")))
    if not files:
        raise SystemExit(f"No *_extraction.csv files in '{folder}'.")
    out_csv = out_csv or os.path.join(folder, "all_samples.csv")

    samples, parameters, conflicts = {}, [], []
    for fn in files:
        with open(fn, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("status") not in KEEP or not (r.get("sample_name") or "").strip():
                    continue
                paper = r.get("paper") or os.path.basename(fn).replace("_extraction.csv", "")
                name = r.get("item") or r.get("parameter")
                value, unit, source = cell_value(r)
                if value is None:
                    continue
                if name not in parameters:
                    parameters.append(name)
                row = samples.setdefault((paper, _norm_name(r["sample_name"])),
                                         {"paper": paper, "doi": r.get("doi", ""),
                                          "sample_name": r["sample_name"].strip(), "is_main": ""})
                if r.get("is_main") == "yes":
                    row["is_main"] = "yes"
                if name in row and row[name] != value:                  # same sample + parameter twice
                    conflicts.append(f"{paper} / {row['sample_name']} / {name}: kept {row[name]}, "
                                     f"ignored {value}")
                    continue
                row[name], row[f"{name}_unit"] = value, unit
                if with_source:
                    row[f"{name}_source"] = source

    columns = list(ID_COLUMNS)
    for p in parameters:
        columns += [p, f"{p}_unit"] + ([f"{p}_source"] if with_source else [])
    rows = sorted(samples.values(), key=lambda r: (r["paper"], r["is_main"] != "yes", r["sample_name"]))
    for r in rows:
        r["is_main"] = r["is_main"] or "no"

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:    # utf-8-sig: Excel shows µ, °
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"Read {len(files)} file(s) -> {out_csv}")
    print(f"  {len(rows)} samples from {len({r['paper'] for r in rows})} papers, {len(parameters)} parameters: "
          + ", ".join(parameters))
    for c in conflicts:
        print(f"  Note: {c}")
    return out_csv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One table for all papers: one row per sample, "
                                                 "one column per parameter.",
                                     epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder with the *_extraction.csv files (e.g. results/)")
    parser.add_argument("-o", "--output", help="Output CSV (default: <folder>/all_samples.csv)")
    parser.add_argument("--with-source", action="store_true",
                        help="Add a <parameter>_source column: value, range average, bound, text")
    args = parser.parse_args(argv)
    organize(args.folder, args.output, args.with_source)


if __name__ == "__main__":
    main()
