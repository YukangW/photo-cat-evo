"""
Combine the extraction results of many papers into ONE table: one row per sample, one column per parameter.

Input:  a folder with <paper>_extraction.csv files (made by `uv run PhotoCatEvo ...`), searched in its
        subfolders too (a batch run puts each paper in results/<paper file>/); --no-recursive: top folder only
Output: one CSV with
    paper | doi | sample_name | <parameter> | <parameter>_unit | <next parameter> | ...
A parameter that never has a unit (e.g. C:N ratio, or text such as the atmosphere) gets no _unit column.

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
    uv run PhotoCatEvo-table results/ --no-recursive     # ignore the subfolders
"""
import argparse
import csv
import glob
import os
import re
import unicodedata

from .units import _NUMBER

ID_COLUMNS = ["paper", "doi", "sample_name"]
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


def find_result_files(folder: str, recursive: bool = True) -> list[str]:
    """All *_extraction.csv files in the folder (and its subfolders, unless recursive=False)."""
    pattern = os.path.join(folder, "**", "*_extraction.csv") if recursive else os.path.join(folder, "*_extraction.csv")
    return sorted(glob.glob(pattern, recursive=recursive))


def organize(folder: str, out_csv: str | None = None, with_source: bool = False, recursive: bool = True) -> str:
    files = find_result_files(folder, recursive)
    if not files:
        where = "or its subfolders " if recursive else ""
        raise SystemExit(f"No *_extraction.csv files in '{folder}' {where}".rstrip() + ".")
    out_csv = out_csv or os.path.join(folder, "all_samples.csv")

    samples, parameters, conflicts = {}, [], []
    files_of_paper = {}                                  # paper -> files it was read from
    for fn in files:
        with open(fn, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("status") not in KEEP or not (r.get("sample_name") or "").strip():
                    continue
                paper = r.get("paper") or os.path.basename(fn).replace("_extraction.csv", "")
                files_of_paper.setdefault(paper, set()).add(fn)
                name = r.get("item") or r.get("parameter")
                value, unit, source = cell_value(r)
                if value is None:
                    continue
                if name not in parameters:
                    parameters.append(name)
                row = samples.setdefault((paper, _norm_name(r["sample_name"])),
                                         {"paper": paper, "doi": r.get("doi", ""),
                                          "sample_name": r["sample_name"].strip(), "_main": False})
                if r.get("is_main") == "yes":
                    row["_main"] = True          # only used to list the main sample first
                if name in row and row[name] != value:                  # same sample + parameter twice
                    conflicts.append(f"{paper} / {row['sample_name']} / {name}: kept {row[name]}, "
                                     f"ignored {value}")
                    continue
                row[name], row[f"{name}_unit"] = value, unit
                if with_source:
                    row[f"{name}_source"] = source

    columns = list(ID_COLUMNS)
    for p in parameters:
        has_unit = any(r.get(f"{p}_unit") for r in samples.values())   # C:N ratio, text values: no unit
        columns += [p] + ([f"{p}_unit"] if has_unit else []) + ([f"{p}_source"] if with_source else [])
    rows = sorted(samples.values(), key=lambda r: (r["paper"], not r["_main"], r["sample_name"]))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:    # utf-8-sig: Excel shows µ, °
        w = csv.DictWriter(f, fieldnames=columns, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Read {len(files)} file(s) -> {out_csv}")
    print(f"  {len(rows)} samples from {len({r['paper'] for r in rows})} papers, {len(parameters)} parameters: "
          + ", ".join(parameters))
    for paper, fns in sorted(files_of_paper.items()):
        if len(fns) > 1:
            print(f"  Warning: paper '{paper}' is in {len(fns)} files ({', '.join(sorted(fns))}); their values "
                  f"were combined (first value kept). Remove old runs or give a narrower folder.")
    for c in conflicts:
        print(f"  Note: {c}")
    return out_csv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One table for all papers: one row per sample, "
                                                 "one column per parameter.",
                                     epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder with the *_extraction.csv files (e.g. results/), "
                                       "subfolders included")
    parser.add_argument("-o", "--output", help="Output CSV (default: <folder>/all_samples.csv)")
    parser.add_argument("--with-source", action="store_true",
                        help="Add a <parameter>_source column: value, range average, bound, text")
    parser.add_argument("--no-recursive", action="store_true", help="Only the folder itself, not its subfolders")
    args = parser.parse_args(argv)
    organize(args.folder, args.output, args.with_source, recursive=not args.no_recursive)


if __name__ == "__main__":
    main()
