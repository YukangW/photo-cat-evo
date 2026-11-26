"""
Collect open-access photocatalysis papers into a spreadsheet and download their PDFs.

How it works
  1. Journals:  the names in journals.json are looked up in OpenAlex; only journals OpenAlex lists as
                open access are searched (the rest are reported and skipped).
  2. Queries:   one search per parameter in the dictionary (parameters.json), e.g.
                (photocatalytic OR photocatalysis OR photocatalyst) AND ("band gap" OR "bandgap" OR ...)
  3. Search:    OpenAlex (open-access articles in those journals, with a PDF link) and arXiv (preprints).
  4. Save:      results/collected_papers.csv and .xlsx, one row per paper (merged with earlier runs).
  5. Download:  each open-access PDF into papers/collected/. The PubMed Central open-access copy is tried
                first (NIH's official dataset, made for automated download), then each open-access copy
                OpenAlex knows (publisher, repositories). Publisher sites often refuse scripts; if every copy
                fails the row says why and keeps the landing page link to open by hand.

Usage:
    python collect_papers.py                                   # all dictionary parameters, OpenAlex + arXiv
    python collect_papers.py --parameters "band gap" "H2 evolution rate" --per-query 10
    python collect_papers.py --from-year 2022 --max-downloads 20
    python collect_papers.py --anywhere                        # any open-access journal, not just journals.json
    python collect_papers.py --no-download                     # spreadsheet only
Then extract data from a downloaded paper:
    python agent.py papers/collected/<file>.pdf "Find the band gap and H2 evolution rate of the best sample"
"""
import argparse
import csv
import json
import os
import re
import time
from datetime import date

import requests

from .parameters import load_parameters
from .sources import USER_AGENT, SourceError, pmc_pdf_url, resolve_journal, search_arxiv, search_openalex

HERE = os.path.dirname(os.path.abspath(__file__))
JOURNALS_FILE = os.path.join(HERE, "journals.json")
CACHE_FILE = os.path.join(".cache", "journals.json")
DEFAULT_OUT = os.path.join("results", "collected_papers")
DEFAULT_PDF_DIR = os.path.join("papers", "collected")
TOPIC_TERMS = ["photocatalytic", "photocatalysis", "photocatalyst"]
MAX_PDF_BYTES = 60 * 1024 * 1024

COLUMNS = ["title", "authors", "year", "journal", "publisher", "doi", "arxiv_id", "source", "oa_status",
           "license", "matched_parameters", "parameters_mentioned", "cited_by", "pdf_file", "download_status",
           "download_source", "pdf_url", "other_pdf_urls", "pmcid", "landing_url", "openalex_id",
           "collected_on", "abstract"]


# ============================================================ 1. journals
def load_journal_names(path: str = JOURNALS_FILE) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [str(n).strip() for n in data.get("journals", []) if str(n).strip()]


def resolve_journals(names: list[str], cache_path: str | None = None, session=None) -> tuple[list[dict], list[str]]:
    """(open-access journals found in OpenAlex, messages about the ones skipped). Lookups are cached."""
    cache_path = cache_path or CACHE_FILE
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    found, skipped = [], []
    for name in names:
        if name not in cache:
            cache[name] = resolve_journal(name, session)
        j = cache[name]
        if j is None:
            skipped.append(f"{name}: not found in OpenAlex")
        elif not j["is_oa"]:
            skipped.append(f"{name}: OpenAlex does not list '{j['display_name']}' as open access")
        else:
            if not j["exact_match"]:
                skipped.append(f"{name}: using closest match '{j['display_name']}'")
            found.append(j)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    return found, skipped


# ============================================================ 2. queries from the dictionary
def search_terms(name: str, entry: dict) -> list[str]:
    """The parameter's name and aliases worth searching for (drops ambiguous short ones like 'Eg')."""
    terms, seen = [], set()
    for t in [name, *entry.get("aliases", [])]:
        t = t.strip()
        if (len(t) >= 4 or (t.isupper() and len(t) >= 3)) and t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)
    return terms


def build_queries(params: dict, only: list[str] | None = None,
                  topic: list[str] | None = None) -> list[dict]:
    """One query per dictionary parameter: {"parameter", "openalex", "arxiv"}.
    Entries with "search_papers": false (e.g. catalyst mass) are skipped unless named in `only`."""
    topic = topic or TOPIC_TERMS
    names = only or [n for n, e in params.items() if e.get("search_papers", True)]
    unknown = [n for n in names if n not in params]
    if unknown:
        raise ValueError(f"Not in the dictionary: {unknown}. See: python parameters.py list")
    queries = []
    for name in names:
        terms = search_terms(name, params[name])
        quoted = ['"' + t + '"' for t in terms]
        queries.append({
            "parameter": name,
            "openalex": "(" + " OR ".join(topic) + ") AND (" + " OR ".join(quoted) + ")",
            "arxiv": "(" + " OR ".join("all:" + t for t in topic) + ") AND ("
                     + " OR ".join("all:" + q for q in quoted) + ")",
        })
    return queries


def parameters_mentioned(record: dict, params: dict) -> list[str]:
    """Dictionary parameters whose name or an alias appears in the title or abstract."""
    text = f"{record.get('title', '')} {record.get('abstract', '')}".lower()
    return [name for name, entry in params.items()
            if any(re.search(rf"(?<!\w){re.escape(t.lower())}(?!\w)", text) for t in search_terms(name, entry))]


# ============================================================ 3. search + merge
def record_key(r: dict) -> str:
    if r.get("doi"):
        return f"doi:{r['doi'].lower()}"
    if r.get("arxiv_id"):
        return "arxiv:" + re.sub(r"v\d+$", "", r["arxiv_id"])
    return "title:" + re.sub(r"[^a-z0-9]+", "", (r.get("title") or "").lower())


def _join(*lists: str) -> str:
    items = []
    for s in lists:
        for x in (s or "").split(";"):
            if x.strip() and x.strip() not in items:
                items.append(x.strip())
    return "; ".join(items)


def merge(existing: list[dict], new: list[dict]) -> list[dict]:
    """Combine rows, one per paper. Earlier download results are kept; matched parameters are combined."""
    by_key = {}
    for r in [*existing, *new]:
        k = record_key(r)
        if k in by_key:
            old = by_key[k]
            by_key[k] = {**r, **{c: v for c, v in old.items() if v not in (None, "")},
                         "matched_parameters": _join(old.get("matched_parameters"), r.get("matched_parameters"))}
        else:
            by_key[k] = r
    return list(by_key.values())


def collect(params: dict, queries: list[dict], journal_ids: list[str] | None, sources: list[str],
            per_query: int = 25, from_year: int | None = None, session=None) -> list[dict]:
    """Run every query on every source. Returns new rows (not yet merged with earlier runs)."""
    rows, today = [], date.today().isoformat()
    for q in queries:
        found = []
        try:
            if "openalex" in sources:
                found += search_openalex(q["openalex"], journal_ids, per_query, from_year, session)
            if "arxiv" in sources:
                found += search_arxiv(q["arxiv"], per_query, session, from_year)
        except SourceError as e:
            print(f"  ! {q['parameter']}: {e}")
            if "budget" in str(e):
                break
        print(f"  {q['parameter']}: {len(found)} paper(s)")
        for r in found:
            rows.append({**r, "matched_parameters": q["parameter"],
                         "parameters_mentioned": "; ".join(parameters_mentioned(r, params)),
                         "pdf_file": "", "download_status": "", "collected_on": today})
    return merge([], rows)


# ============================================================ 4. spreadsheet
def read_spreadsheet(csv_path: str) -> list[dict]:
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def write_spreadsheet(rows: list[dict], out_base: str) -> list[str]:
    """Write <out_base>.csv and, if openpyxl is installed, <out_base>.xlsx. Returns the files written."""
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    rows = sorted(rows, key=lambda r: (-(int(r.get("year") or 0)), r.get("title") or ""))
    with open(f"{out_base}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows({c: "" if r.get(c) is None else r.get(c) for c in COLUMNS} for r in rows)
    written = [f"{out_base}.csv"]
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        return written
    wb = Workbook()
    ws = wb.active
    ws.title = "papers"
    ws.append(COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in rows:
        ws.append(["" if r.get(c) is None else str(r.get(c))[:32000] for c in COLUMNS])
    widths = {"title": 60, "authors": 30, "journal": 25, "abstract": 80, "pdf_url": 40, "landing_url": 40,
              "matched_parameters": 30, "parameters_mentioned": 30, "pdf_file": 35}
    for i, c in enumerate(COLUMNS, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(c, 14)
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(f"{out_base}.xlsx")
    return [*written, f"{out_base}.xlsx"]


# ============================================================ 5. download
def pdf_filename(r: dict) -> str:
    base = r.get("doi") or (f"arXiv_{r['arxiv_id']}" if r.get("arxiv_id") else r.get("title", "paper")[:80])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") + ".pdf"


def _fetch_pdf(url: str, session) -> tuple[bytes | None, str]:
    """(PDF bytes, '') or (None, why it failed)."""
    try:
        resp = session.get(url, timeout=90, stream=True, allow_redirects=True,
                           headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"
        chunks, size = [], 0
        for chunk in resp.iter_content(64 * 1024):
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_PDF_BYTES:
                return None, "too_large"
    except requests.RequestException as e:
        return None, f"error: {type(e).__name__}"
    data = b"".join(chunks)
    if not data.startswith(b"%PDF"):
        return None, "not_pdf"
    return data, ""


def _source_name(url: str) -> str:
    host = re.sub(r"^https?://(www\.)?", "", url).split("/", 1)[0]
    return "PubMed Central (open-access data)" if "pmc-oa-opendata" in host else host


def download_pdf(r: dict, folder: str, session=None) -> tuple[str, str, str]:
    """Download one open-access PDF, trying each known copy in turn.
    Returns (file path or '', status, where it came from)."""
    path = os.path.join(folder, pdf_filename(r))
    if os.path.exists(path):
        return path, "already_downloaded", r.get("download_source") or ""
    session = session or requests.Session()
    candidates = []
    pmc = pmc_pdf_url(r.get("pmcid") or "", session) if r.get("pmcid") else None
    for url in [pmc, r.get("pdf_url"), *(r.get("other_pdf_urls") or "").split()]:
        if url and url not in candidates:
            candidates.append(url)
    if not candidates:
        return "", "no_pdf_link", ""
    failures = []
    for url in candidates:
        data, why = _fetch_pdf(url, session)
        if data:
            os.makedirs(folder, exist_ok=True)
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return path, "downloaded", _source_name(url)
        failures.append(f"{_source_name(url)} {why}")
    hint = " (publisher refused the script; open landing_url by hand)" if any(
        "http_403" in f or "not_pdf" in f for f in failures) else ""
    return "", "failed: " + "; ".join(failures) + hint, ""


def download_all(rows: list[dict], folder: str, max_downloads: int | None = None, session=None,
                 delay_s: float = 1.0) -> list[dict]:
    """Download PDFs for rows that do not have one yet. Returns the updated rows."""
    out, done = [], 0
    for r in rows:
        if r.get("download_status") in ("downloaded", "already_downloaded") and r.get("pdf_file") \
                and os.path.exists(r["pdf_file"]):
            out.append(r)
            continue
        if max_downloads is not None and done >= max_downloads:
            out.append(r)
            continue
        path, status, source = download_pdf(r, folder, session)
        done += 1
        shown = f"downloaded from {source}" if status == "downloaded" else status
        print(f"  [{done}] {(r.get('title') or '')[:60]}\n        {shown}")
        out.append({**r, "pdf_file": path, "download_status": status, "download_source": source})
        time.sleep(3.0 if "arxiv.org" in (r.get("pdf_url") or "") else delay_s)
    return out


# ============================================================ command line
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Collect open-access photocatalysis papers using the parameter "
                                            "dictionary, save them to a spreadsheet and download the PDFs.",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parameters", nargs="+", help="Dictionary parameters to search for (default: all)")
    p.add_argument("--topic", nargs="+", default=TOPIC_TERMS, help="Topic words (default: photocatalytic ...)")
    p.add_argument("--sources", nargs="+", default=["openalex", "arxiv"], choices=["openalex", "arxiv"])
    p.add_argument("--journals", default=JOURNALS_FILE, help="Journal list (default journals.json)")
    p.add_argument("--anywhere", action="store_true", help="Search all open-access journals, not just the list")
    p.add_argument("--per-query", type=int, default=25, help="Papers per parameter per source (default 25)")
    p.add_argument("--from-year", type=int, help="Only papers from this year on")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"Spreadsheet path without extension (default {DEFAULT_OUT})")
    p.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR, help=f"Folder for PDFs (default {DEFAULT_PDF_DIR})")
    p.add_argument("--no-download", action="store_true", help="Only make the spreadsheet")
    p.add_argument("--max-downloads", type=int, help="Download at most this many PDFs this run")
    args = p.parse_args(argv)

    params = load_parameters()
    try:
        queries = build_queries(params, args.parameters, args.topic)
    except ValueError as e:
        raise SystemExit(str(e))

    journal_ids = None
    if "openalex" in args.sources and not args.anywhere:
        print("Checking journals in OpenAlex...")
        journals, skipped = resolve_journals(load_journal_names(args.journals))
        for msg in skipped:
            print(f"  - {msg}")
        print(f"  {len(journals)} open-access journal(s) will be searched")
        journal_ids = [j["id"] for j in journals]
        if not journal_ids:
            raise SystemExit("No open-access journals found. Check journals.json or use --anywhere.")

    print(f"\nSearching {' + '.join(args.sources)} for {len(queries)} parameter(s)...")
    new_rows = collect(params, queries, journal_ids, args.sources, args.per_query, args.from_year)
    rows = merge(read_spreadsheet(f"{args.out}.csv"), new_rows)
    files = write_spreadsheet(rows, args.out)
    print(f"\n{len(new_rows)} paper(s) found this run, {len(rows)} in the spreadsheet: {', '.join(files)}")

    if not args.no_download:
        print(f"\nDownloading PDFs to {args.pdf_dir}/ ...")
        rows = download_all(rows, args.pdf_dir, args.max_downloads)
        write_spreadsheet(rows, args.out)
        ok = sum(r.get("download_status") in ("downloaded", "already_downloaded") for r in rows)
        print(f"\n{ok} of {len(rows)} PDF(s) in {args.pdf_dir}/. Spreadsheet updated.")


if __name__ == "__main__":
    main()
