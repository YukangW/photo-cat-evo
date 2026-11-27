"""
Evaluate the extraction against a hand-made ground truth, in the style of Li et al. (Adv. Sci. 2026).

Ground truth (CSV or .xlsx), one row per sample, columns as in the team's sheet:
    Paper | DOI | ... | Sample Name | Specific Surface Area | Specific Surface Area Units | Band Gap | ...
    | C:N ratio | Hydrogen Evolution Rate | Hydrogen Evolution Rate Units | ... | Atmosphere used | Notes
Predictions: the merged table made by `uv run PhotoCatEvo-table results/` (results/all_samples.csv):
    paper | doi | sample_name | band gap | band gap_unit | BET surface area | ...

How it works
  1. Papers are matched by DOI (the paper names differ between the two files).
     Only papers that are in BOTH files are scored; the others are listed.
  2. Samples are matched inside each paper by name: case, spaces, dashes and subscripts are ignored,
     then close spellings (fuzzy), then a cocatalyst prefix/suffix ("Pt-CNU20" ~ "CNU20").
  3. Columns are matched through the parameter dictionary ("Specific Surface Area" -> "BET surface area").
  4. Before comparing, the predicted value is converted to the ground truth's unit with Pint.
     Numbers are correct within a tolerance: 2 % (--rel-tol), but 0.02 eV for the band gap (ABS_TOL);
     text is compared normalised ("nitrogen" = "N2").
  5. Labels per value (paper, Section 4.2):
        TP  correct value
        FP  wrong value, or a value where the ground truth has none (for a sample of the ground truth)
        FN  value in the ground truth but not extracted (or the sample was not found)
        TN  both empty (not used in precision / recall)
  6. Per parameter: precision = TP/(TP+FP), recall = TP/(TP+FN), F1; plus all parameters together.

Only the papers of the ground truth are scored: extra extracted papers are listed and ignored.
Within those papers, only the samples of the ground truth are scored: extra predicted samples
(not in the ground truth) are listed in matches.csv and ignored. Use --score-extra to count their values as FP.
To get that subset as a file of its own (to check it, or to analyse only those papers):
    uv run PhotoCatEvo-filter ground_truth.csv results/all_samples.csv      -> results/all_samples_gt_papers.csv

Usage
    uv run PhotoCatEvo-evaluate ground_truth.csv results/all_samples.csv
    uv run PhotoCatEvo-evaluate ground_truth.xlsx results/all_samples.csv --out evaluation/
    uv run PhotoCatEvo-evaluate gt.csv pred.csv --strict        # a wrong value counts as FP and FN
    uv run PhotoCatEvo-evaluate gt.csv pred.csv --rel-tol 0.05  # 5 % tolerance for numbers
Outputs (in --out, default evaluation/):
    metrics.csv   TP/FP/FN/TN, precision, recall, F1 per parameter
    details.csv   every comparison with its label (read this to see the errors)
    matches.csv   which predicted sample was matched to which true sample, and how
"""
import argparse
import csv
import difflib
import os
import re
import unicodedata
from collections import defaultdict

from .parameters import find_parameter, load_parameters
from .units import UnitError, convert

NA = {"", "na", "n/a", "none", "null", "nan", "-", "not reported"}
ABS_TOL = {"band gap": 0.02}             # absolute tolerances (in the ground truth's unit)
FUZZY = 0.85

# Columns of the ground truth that are not parameters
GT_ID = {"paper", "doi", "sample name", "sample", "sample_name", "notes", "note",
         "number of c3n4 photocatalysts discussed"}

SYNONYMS = {"nitrogen": "n2", "argon": "ar", "ammonia": "nh3", "hydrogen": "h2", "air atmosphere": "air",
            "static air": "air", "teoa": "triethanolamine", "meoh": "methanol"}

_COCAT = r"(?:Pt|Au|Ag|Pd|Rh|Ru|Ir|Ni|Co|Cu|NiS|CoP|Ni2P|MoS2)"
_LOAD = r"(?:\d+(?:\.\d+)?\s*(?:wt\s*%|wt%|%|mol\s*%)\s*)?"
_SEP = r"[\s\-‐-―−/@]"


# ============================================================ normalising
def norm_doi(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).strip().lower()
    s = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", s)
    return re.sub(r"\s+", "", s).rstrip(".")


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    return re.sub(r"[\s\-‐-―−_]+", "", s)


def strip_cocatalyst(name: str) -> str:
    """'Pt-CNU20' -> 'CNU20', '0.5 wt% Pt/TiO2' -> 'TiO2'; unchanged if there is no cocatalyst."""
    n = unicodedata.normalize("NFKC", name or "").strip()
    m = re.match(rf"^{_LOAD}{_COCAT}{_SEP}+(.+)$", n) or re.match(rf"^(.+?){_SEP}+{_LOAD}{_COCAT}$", n)
    return m.group(1).strip() if m else n


def norm_text(s: str) -> str:
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).strip().lower())
    return SYNONYMS.get(s, s)


def is_na(v) -> bool:
    return v is None or norm_text(v) in NA


def as_number(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = unicodedata.normalize("NFKC", str(v)).strip().replace("−", "-")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    try:
        return float(s)
    except ValueError:
        return None


# ============================================================ loading
def _read_table(path: str) -> list[dict]:
    if path.lower().endswith((".xlsx", ".xlsm")):
        try:
            import openpyxl
        except ImportError as e:
            raise SystemExit("Reading .xlsx needs openpyxl (uv add openpyxl), or export the sheet as CSV.") from e
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        rows = list(ws.iter_rows(values_only=True))
        header = [str(h or "").strip() for h in rows[0]]
        return [{h: ("" if v is None else v) for h, v in zip(header, r)} for r in rows[1:]]
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _find(row_keys, *names):
    low = {k.strip().lower(): k for k in row_keys}
    return next((low[n] for n in names if n in low), None)


def load_truth(path: str, params: dict) -> tuple[list[dict], dict]:
    """-> entries [{doi, paper, sample, values{param: (value, unit)}}], {param: ground-truth column}"""
    rows = _read_table(path)
    if not rows:
        raise SystemExit(f"{path} is empty.")
    keys = list(rows[0].keys())
    c_doi, c_paper = _find(keys, "doi"), _find(keys, "paper")
    c_sample = _find(keys, "sample name", "sample_name", "sample")
    if not c_doi or not c_sample:
        raise SystemExit(f"{path} needs a DOI column and a Sample Name column.")
    columns = {}
    for k in keys:
        low = k.strip().lower()
        if low in GT_ID or low.endswith(" units") or low.endswith(" unit") or low.endswith("_unit"):
            continue
        match = find_parameter(k, params)
        columns[match[0] if match else k.strip()] = k
    entries = []
    for r in rows:
        doi, sample = norm_doi(r.get(c_doi)), str(r.get(c_sample) or "").strip()
        if not doi or not sample:
            continue
        values = {}
        for param, col in columns.items():
            unit_col = _find(keys, f"{col.strip().lower()} units", f"{col.strip().lower()} unit")
            v = r.get(col)
            values[param] = (None if is_na(v) else v, str(r.get(unit_col) or "").strip() if unit_col else "")
        entries.append({"doi": doi, "paper": str(r.get(c_paper) or ""), "sample": sample, "values": values})
    return entries, columns


def load_predictions(path: str, params: dict, skipped: list | None = None) -> list[dict]:
    rows = _read_table(path)
    entries = []
    for r in rows:
        doi, sample = norm_doi(r.get("doi")), str(r.get("sample_name") or "").strip()
        if not doi or not sample:
            if skipped is not None and not doi:
                skipped.append(str(r.get("paper") or "?"))
            continue
        values = {}
        for k, v in r.items():
            if k in ("paper", "doi", "sample_name", "is_main") or k.endswith("_unit") or k.endswith("_source"):
                continue
            match = find_parameter(k, params)
            values[match[0] if match else k] = (None if is_na(v) else v, str(r.get(f"{k}_unit") or "").strip())
        entries.append({"doi": doi, "paper": r.get("paper", ""), "sample": sample, "values": values})
    return entries


# ============================================================ matching
def merge_loaded(pred: list[dict], truth_names: set[str]) -> list[dict]:
    """In one paper, fold 'Pt-CNU20' into 'CNU20' when the ground truth only has CNU20
    (the activity is often measured on the loaded sample, the properties on the parent)."""
    by_name = {norm_name(p["sample"]): p for p in pred}
    keep = []
    for p in pred:
        n = norm_name(p["sample"])
        parent = norm_name(strip_cocatalyst(p["sample"]))
        if parent != n and n not in truth_names and parent in by_name:
            target = by_name[parent]
            for k, v in p["values"].items():
                if target["values"].get(k, (None, ""))[0] is None and v[0] is not None:
                    target["values"][k] = v
                    target.setdefault("merged_from", set()).add(p["sample"])
            continue
        keep.append(p)
    return keep


def match_samples(truth: list[dict], pred: list[dict]) -> list[tuple]:
    pairs, t_left, p_left = [], list(range(len(truth))), list(range(len(pred)))

    def take(ti, pi, how):
        pairs.append((ti, pi, how))
        t_left.remove(ti)
        p_left.remove(pi)

    for ti in list(t_left):                                         # 1. same name
        for pi in p_left:
            if norm_name(truth[ti]["sample"]) == norm_name(pred[pi]["sample"]):
                take(ti, pi, "exact")
                break
    for ti in list(t_left):                                         # 2. cocatalyst stripped
        for pi in p_left:
            if norm_name(strip_cocatalyst(truth[ti]["sample"])) == norm_name(strip_cocatalyst(pred[pi]["sample"])):
                take(ti, pi, "cocatalyst")
                break
    scored = sorted(((difflib.SequenceMatcher(None, norm_name(truth[ti]["sample"]),
                                              norm_name(pred[pi]["sample"])).ratio(), ti, pi)
                     for ti in t_left for pi in p_left), reverse=True)
    for score, ti, pi in scored:                                    # 3. close spelling
        if score >= FUZZY and ti in t_left and pi in p_left:
            take(ti, pi, f"fuzzy {score:.2f}")
    pairs += [(ti, None, "missed") for ti in t_left]
    pairs += [(None, pi, "extra") for pi in p_left]
    return pairs


# ============================================================ scoring
def same_value(param: str, truth, pred, rel_tol: float) -> tuple[bool, str]:
    """Is the predicted (value, unit) equal to the true (value, unit)? Returns (equal, note)."""
    (tv, tu), (pv, pu) = truth, pred
    tn, pn = as_number(tv), as_number(pv)
    if tn is None or pn is None:                                    # text, e.g. the atmosphere
        return norm_text(tv) == norm_text(pv), ""
    note = ""
    if tu and pu and norm_text(tu) != norm_text(pu):
        try:
            before = pn
            pn = convert(pn, pu, tu)
            if abs(pn - before) > 1e-9 * max(1.0, abs(before)):     # only when the number really changed
                note = f"converted {pv} {pu} -> {pn:g} {tu}"
        except (UnitError, ValueError) as e:
            note = f"units not comparable ({e}); raw numbers compared"
    # a fixed tolerance where one is defined (band gap: 0.02 eV), otherwise relative (default 2 %)
    tol = ABS_TOL[param] if param in ABS_TOL else rel_tol * abs(tn)
    return abs(pn - tn) <= tol + 1e-12, note


def label(param, truth, pred, rel_tol, strict) -> tuple[list[str], str]:
    t_empty, p_empty = truth[0] is None, pred[0] is None
    if t_empty and p_empty:
        return ["TN"], ""
    if t_empty:
        return ["FP"], "not in the ground truth"
    if p_empty:
        return ["FN"], "not extracted"
    ok, note = same_value(param, truth, pred, rel_tol)
    if ok:
        return ["TP"], note
    return (["FP", "FN"] if strict else ["FP"]), note or "different value"


def evaluate(truth: list[dict], params_cols: list[str], pred: list[dict], rel_tol=0.02, strict=False,
             merge=True, score_extra=False) -> dict:
    t_by, p_by = defaultdict(list), defaultdict(list)
    for e in truth:
        t_by[e["doi"]].append(e)
    for e in pred:
        p_by[e["doi"]].append(e)
    common = sorted(set(t_by) & set(p_by))
    details, matches = [], []
    for doi in common:
        T, P = t_by[doi], p_by[doi]
        if merge:
            P = merge_loaded(P, {norm_name(t["sample"]) for t in T})
        for ti, pi, how in match_samples(T, P):
            t, p = (T[ti] if ti is not None else None), (P[pi] if pi is not None else None)
            matches.append({"doi": doi, "paper": (t or p)["paper"], "true_sample": t["sample"] if t else "",
                            "predicted_sample": p["sample"] if p else "", "match": how,
                            "merged_from": ", ".join(sorted(p.get("merged_from", []))) if p else ""})
            if t is None and not score_extra:                       # extra predicted sample: not scored
                continue
            for param in params_cols:
                tv = t["values"].get(param, (None, "")) if t else (None, "")
                pv = p["values"].get(param, (None, "")) if p else (None, "")
                labels, note = label(param, tv, pv, rel_tol, strict)
                for lab in labels:
                    details.append({"doi": doi, "paper": (t or p)["paper"],
                                    "true_sample": t["sample"] if t else "",
                                    "predicted_sample": p["sample"] if p else "", "parameter": param,
                                    "true_value": "" if tv[0] is None else tv[0], "true_unit": tv[1],
                                    "predicted_value": "" if pv[0] is None else pv[0], "predicted_unit": pv[1],
                                    "label": lab, "note": note})
    metrics = []
    for param in params_cols + ["ALL"]:
        c = defaultdict(int)
        for d in details:
            if param == "ALL" or d["parameter"] == param:
                c[d["label"]] += 1
        prec = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else None
        rec = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else None
        f1 = (2 * prec * rec / (prec + rec) if prec + rec else 0.0) if prec is not None and rec is not None else None
        metrics.append({"parameter": param, "TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"],
                        "precision": prec, "recall": rec, "F1": f1})
    return {"score_extra": score_extra, "metrics": metrics, "details": details, "matches": matches, "papers": common,
            "only_truth": sorted(set(t_by) - set(p_by)), "only_pred": sorted(set(p_by) - set(t_by))}


# ============================================================ filter step
def filter_to_truth(truth_path: str, pred_path: str, out_csv: str | None = None,
                    params: dict | None = None, verbose: bool = False) -> dict:
    """Write a copy of the predictions table with only the papers that are in the ground truth
    (matched by DOI). All columns are kept as they are. Returns a summary."""
    params = load_parameters() if params is None else params
    truth, _ = load_truth(truth_path, params)
    truth_dois = {e["doi"] for e in truth}
    with open(pred_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    if not header or "doi" not in header:
        raise SystemExit(f"{pred_path} has no 'doi' column: is it the table made by PhotoCatEvo-table?")

    kept, dropped, no_doi = [], set(), set()
    for r in rows:
        doi = norm_doi(r.get("doi"))
        if not doi:
            no_doi.add(r.get("paper") or "?")
        elif doi in truth_dois:
            kept.append(r)
        else:
            dropped.add(r.get("paper") or doi)
    found = {norm_doi(r["doi"]) for r in kept}
    names = {e["doi"]: e["paper"] or e["doi"] for e in truth}
    if verbose:
        print(f"Ground truth: {len(truth_dois)} paper(s). Predictions: {len(rows)} row(s).")
        for doi in sorted(truth_dois, key=lambda d: names[d]):
            mine = [r for r in kept if norm_doi(r["doi"]) == doi]
            if mine:
                samples = ", ".join(r.get("sample_name", "?") for r in mine)
                print(f"  KEEP  {names[doi]:<14} {doi:<38} <- {mine[0].get('paper')}: {len(mine)} sample(s): {samples}")
            else:
                print(f"  MISS  {names[doi]:<14} {doi:<38} (no extraction with this DOI)")
        for paper in sorted(dropped):
            print(f"  DROP  {paper} (its DOI is not in the ground truth)")
        for paper in sorted(no_doi):
            print(f"  ????  {paper} (no DOI: cannot be matched)")
    summary = {"kept_papers": len(found), "kept_rows": len(kept), "dropped_papers": sorted(dropped),
               "papers_without_doi": sorted(no_doi),
               "truth_papers_not_extracted": sorted(names[d] for d in truth_dois - found)}

    out_csv = out_csv or os.path.splitext(pred_path)[0] + "_gt_papers.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(kept)
    summary["output"] = out_csv
    return summary


def filter_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Keep only the papers of the ground truth in a predictions table.")
    ap.add_argument("truth", help="Ground truth: CSV or .xlsx with a DOI column")
    ap.add_argument("predictions", help="Table made by PhotoCatEvo-table (all_samples.csv)")
    ap.add_argument("-o", "--output", help="Output CSV (default: <predictions>_gt_papers.csv)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print every paper: kept, missing or dropped")
    a = ap.parse_args(argv)
    s = filter_to_truth(a.truth, a.predictions, a.output, verbose=a.verbose)
    print(f"Kept {s['kept_papers']} paper(s) of the ground truth ({s['kept_rows']} samples) -> {s['output']}")
    print(f"Left out {len(s['dropped_papers'])} extracted paper(s) that are not in the ground truth.")
    if s["truth_papers_not_extracted"]:
        print(f"Ground-truth papers with no extraction: {', '.join(s['truth_papers_not_extracted'])}")
    if s["papers_without_doi"]:
        print(f"Warning: no DOI, so they could not be matched: {', '.join(s['papers_without_doi'])}. "
              f"Run them again with --doi, or add the DOI to the table.")


# ============================================================ output
def _write(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


SYMBOL = {"TP": "OK  ", "FP": "WRONG", "FN": "MISSED", "TN": ""}


def print_details(res: dict, columns) -> None:
    """Every paper, how its samples were paired, and every comparison (TN = both empty, not shown)."""
    print("\n" + "=" * 100 + "\nDETAILS (OK = TP, WRONG = FP, MISSED = FN; both empty is not shown)\n" + "=" * 100)
    for doi in res["papers"]:
        matches = [m for m in res["matches"] if m["doi"] == doi]
        print(f"\nPaper {matches[0]['paper'] if matches else ''}  ({doi})")
        for m in matches:
            merged = f"  (+ values of {m['merged_from']})" if m["merged_from"] else ""
            tag = "extra, not scored" if m["match"] == "extra" and not res.get("score_extra") else m["match"]
            print(f"  sample  truth: {m['true_sample'] or '-':<18} prediction: {m['predicted_sample'] or '-':<18} "
                  f"[{tag}]{merged}")
        for d in res["details"]:
            if d["doi"] != doi or d["label"] == "TN":
                continue
            sample = d["true_sample"] or f"(extra) {d['predicted_sample']}"
            truth = f"{d['true_value']} {d['true_unit']}".strip() or "-"
            pred = f"{d['predicted_value']} {d['predicted_unit']}".strip() or "-"
            note = f"   ({d['note']})" if d["note"] and d["label"] == "TP" and "converted" in d["note"] else ""
            print(f"    {SYMBOL[d['label']]:<6} {sample:<18} {d['parameter']:<22} truth: {truth:<18} "
                  f"predicted: {pred}{note}")
    print()


def _fmt(x):
    return "  -" if x is None else f"{x:.2f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Precision / recall / F1 of the extraction against a ground truth.",
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth", help="Ground truth: CSV or .xlsx, one row per sample")
    ap.add_argument("predictions", help="Merged predictions from PhotoCatEvo-table (all_samples.csv)")
    ap.add_argument("--out", default="evaluation", help="Output folder (default evaluation/)")
    ap.add_argument("--rel-tol", type=float, default=0.02, help="Relative tolerance for numbers (default 0.02)")
    ap.add_argument("--strict", action="store_true", help="Count a wrong value as FP and FN")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print every paper, every sample match and every comparison")
    ap.add_argument("--score-extra", action="store_true",
                    help="Also score predicted samples that are not in the ground truth (their values count as FP)")
    ap.add_argument("--no-merge-loaded", action="store_true",
                    help="Do not fold 'Pt-X' predictions into sample X of the ground truth")
    a = ap.parse_args(argv)

    params = load_parameters()
    truth, columns = load_truth(a.truth, params)
    no_doi = []
    pred = load_predictions(a.predictions, params, skipped=no_doi)
    res = evaluate(truth, list(columns), pred, a.rel_tol, a.strict, not a.no_merge_loaded, a.score_extra)

    print(f"Ground truth: {len(truth)} samples in {len({e['doi'] for e in truth})} papers; "
          f"parameters: {', '.join(columns)}")
    print(f"Scored {len(res['papers'])} paper(s) found in both files (matched by DOI).")
    if res["only_truth"]:
        print(f"  Not in the predictions (not scored): {', '.join(res['only_truth'])}")
    if res["only_pred"]:
        print(f"  Extracted but not in the ground truth (not scored): {len(res['only_pred'])} paper(s)")
    if no_doi:
        print(f"  Warning: {len(set(no_doi))} extracted paper(s) have no DOI and could not be matched: "
              f"{', '.join(sorted(set(no_doi)))}")
    print(f"Labels as in Li et al.{' (strict: wrong value = FP + FN)' if a.strict else ' (wrong value = FP)'}, "
          f"tolerance {a.rel_tol:.0%}\n")
    print(f"{'parameter':<22}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'prec':>7}{'recall':>8}{'F1':>7}")
    for m in res["metrics"]:
        print(f"{m['parameter'][:21]:<22}{m['TP']:>4}{m['FP']:>4}{m['FN']:>4}{m['TN']:>4}"
              f"{_fmt(m['precision']):>7}{_fmt(m['recall']):>8}{_fmt(m['F1']):>7}")
    how = defaultdict(int)
    for mt in res["matches"]:
        how[mt["match"].split()[0]] += 1
    print("\nSample matching: " + ", ".join(f"{k}={v}" for k, v in sorted(how.items())))
    if how.get("extra"):
        print(f"  {how['extra']} extra predicted sample(s) not in the ground truth: "
              + ("scored as FP (--score-extra)" if a.score_extra else "not scored (use --score-extra to count them)"))

    if a.verbose:
        print_details(res, columns)

    os.makedirs(a.out, exist_ok=True)
    _write(os.path.join(a.out, "metrics.csv"), res["metrics"])
    _write(os.path.join(a.out, "details.csv"), res["details"])
    _write(os.path.join(a.out, "matches.csv"), res["matches"])
    print(f"Saved metrics.csv, details.csv and matches.csv in {a.out}/")


if __name__ == "__main__":
    main()
