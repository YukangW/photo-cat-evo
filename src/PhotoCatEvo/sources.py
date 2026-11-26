"""
Where open-access papers are found. Plain HTTP, no LLM.

  OpenAlex (https://openalex.org)  catalogue of ~250M works. Used to check which journals are open access and
                                   to search them; gives the PDF link of the open-access version.
                                   Free; without a key all users on your network share a small daily budget
                                   (~100 requests). For more, get a free key yourself at
                                   https://openalex.org/settings/api and `export OPENALEX_API_KEY=...`.
  arXiv (https://arxiv.org)        preprints (cond-mat, physics.chem-ph, ...); every paper has a free PDF.
  PubMed Central open-access data  NIH's official copy of open-access papers (RSC Advances, MDPI, Sci. Rep.,
  (pmc-oa-opendata on AWS)         ACS Omega, ... deposit there). Made for automated download, so it works when
                                   publisher sites refuse scripts.

Every function takes an optional `session` (anything with requests' .get) so tests can run offline.
"""
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

OPENALEX = "https://api.openalex.org"
ARXIV = "https://export.arxiv.org/api/query"
PMC_OA = "https://pmc-oa-opendata.s3.amazonaws.com"
# A generic client name only: no personal details are sent to these services.
USER_AGENT = "paper-collector/0.1 (open-access research tool)"
ARXIV_DELAY_S = 3.0          # arXiv asks for at most one request every 3 seconds

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


class SourceError(RuntimeError):
    """A search service could not be reached or refused the request."""


def _get(url: str, params: dict, session=None, retries: int = 3):
    session = session or _session
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            err = f"{type(e).__name__}"
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429 and "openalex" in url:
                raise SourceError("OpenAlex's free daily budget for your network is used up. Try again tomorrow, "
                                  "or get a free key at https://openalex.org/settings/api and "
                                  "`export OPENALEX_API_KEY=...`.")
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise SourceError(f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}")
            err = f"HTTP {resp.status_code}"
        time.sleep(3 * 2 ** attempt)
    raise SourceError(f"{url} failed after {retries} tries ({err}).")


def _openalex_params(params: dict) -> dict:
    key = os.getenv("OPENALEX_API_KEY")
    return {**params, "api_key": key} if key else params


def _bare_doi(doi: str | None) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi or "").lower()


# ============================================================ OpenAlex: journals
def resolve_journal(name: str, session=None) -> dict | None:
    """Look a journal up by name. Returns its OpenAlex id, ISSN and open-access status, or None."""
    resp = _get(f"{OPENALEX}/sources", _openalex_params({
        "search": name, "per-page": 5,
        "select": "id,display_name,issn_l,is_oa,is_in_doaj,host_organization_name,type"}), session)
    results = [r for r in resp.json().get("results", []) if r.get("type") in (None, "journal")]
    if not results:
        return None
    exact = [r for r in results if r["display_name"].lower() == name.lower()]
    r = (exact or results)[0]
    return {"name": name, "id": r["id"].rsplit("/", 1)[-1], "display_name": r["display_name"],
            "issn_l": r.get("issn_l") or "", "is_oa": bool(r.get("is_oa")),
            "is_in_doaj": bool(r.get("is_in_doaj")), "publisher": r.get("host_organization_name") or "",
            "exact_match": bool(exact)}


# ============================================================ OpenAlex: works
def _abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted:
        return ""
    words = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(word for _, word in words)


def _authors(names: list[str]) -> str:
    return "; ".join(names[:5]) + ("; et al." if len(names) > 5 else "")


def _openalex_record(w: dict) -> dict:
    best = w.get("best_oa_location") or {}
    primary = w.get("primary_location") or {}
    source = (primary.get("source") or best.get("source") or {})
    pdf_urls = []
    for loc in [best, primary, *(w.get("locations") or [])]:
        url = loc.get("pdf_url")
        if loc.get("is_oa", True) and url and url not in pdf_urls:
            pdf_urls.append(url)
    pmcid = ((w.get("ids") or {}).get("pmcid") or "").rstrip("/").rsplit("/", 1)[-1]
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    return {
        "title": w.get("title") or "",
        "authors": _authors([a["author"]["display_name"] for a in w.get("authorships", [])
                             if a.get("author", {}).get("display_name")]),
        "year": w.get("publication_year"),
        "journal": source.get("display_name") or "",
        "publisher": source.get("host_organization_name") or "",
        "doi": _bare_doi(w.get("doi")),
        "arxiv_id": "",
        "openalex_id": (w.get("id") or "").rsplit("/", 1)[-1],
        "oa_status": (w.get("open_access") or {}).get("oa_status") or "",
        "license": best.get("license") or primary.get("license") or "",
        "pdf_url": pdf_urls[0] if pdf_urls else "",
        "other_pdf_urls": " ".join(pdf_urls[1:]),
        "pmcid": pmcid,
        "landing_url": best.get("landing_page_url") or primary.get("landing_page_url") or "",
        "cited_by": w.get("cited_by_count"),
        "abstract": _abstract(w.get("abstract_inverted_index")),
        "source": "OpenAlex",
    }


def search_openalex(query: str, source_ids: list[str] | None = None, per_page: int = 25,
                    from_year: int | None = None, session=None) -> list[dict]:
    """Open-access works with a PDF link matching the query (OpenAlex search syntax: AND / OR / "phrases"),
    optionally only in the given journals (OpenAlex source ids, e.g. 'S2481244646' = RSC Advances)."""
    filters = ["is_oa:true", "has_pdf_url:true", "type:article|preprint|review"]
    if source_ids:
        filters.append("primary_location.source.id:" + "|".join(source_ids[:100]))
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    resp = _get(f"{OPENALEX}/works", _openalex_params({
        "search": query, "filter": ",".join(filters), "per-page": min(per_page, 200), "sort": "relevance_score:desc",
        "select": "id,ids,doi,title,publication_year,primary_location,best_oa_location,locations,"
                  "open_access,authorships,cited_by_count,abstract_inverted_index"}), session)
    return [_openalex_record(w) for w in resp.json().get("results", [])]


# ============================================================ arXiv
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _arxiv_record(e: ET.Element) -> dict:
    text = lambda tag: " ".join((e.findtext(tag, "", _NS) or "").split())  # noqa: E731
    arxiv_id = text("a:id").rsplit("/abs/", 1)[-1]
    pdf = next((ln.get("href") for ln in e.findall("a:link", _NS) if ln.get("title") == "pdf"),
               f"https://arxiv.org/pdf/{arxiv_id}")
    return {
        "title": text("a:title"),
        "authors": _authors([a.findtext("a:name", "", _NS) for a in e.findall("a:author", _NS)]),
        "year": int(text("a:published")[:4]) if text("a:published") else None,
        "journal": text("arxiv:journal_ref") or "arXiv preprint",
        "publisher": "arXiv",
        "doi": text("arxiv:doi").lower(),
        "arxiv_id": arxiv_id,
        "openalex_id": "",
        "oa_status": "green",
        "license": "see arXiv abstract page",
        "pdf_url": pdf,
        "other_pdf_urls": "",
        "pmcid": "",
        "landing_url": f"https://arxiv.org/abs/{arxiv_id}",
        "cited_by": None,
        "abstract": text("a:summary"),
        "source": "arXiv",
    }


def search_arxiv(query: str, max_results: int = 25, session=None, from_year: int | None = None) -> list[dict]:
    """arXiv preprints matching an arXiv query, e.g. 'all:photocatalytic AND all:"band gap"'."""
    if from_year:
        query = f"({query}) AND submittedDate:[{from_year}01010000 TO 299912312359]"
    resp = _get(ARXIV, {"search_query": query, "max_results": max_results, "sortBy": "relevance"}, session)
    time.sleep(0 if session else ARXIV_DELAY_S)
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        raise SourceError(f"arXiv returned something that is not XML ({e}).") from e
    return [_arxiv_record(e) for e in root.findall("a:entry", _NS)]


# ============================================================ PubMed Central open-access data
def pmc_pdf_url(pmcid: str, session=None) -> str | None:
    """URL of the PDF of the newest version of a PMC open-access paper, or None if it is not in the dataset."""
    if not re.fullmatch(r"PMC\d+", pmcid or ""):
        return None
    try:
        resp = _get(PMC_OA, {"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"}, session, retries=2)
    except SourceError:
        return None
    versions = re.findall(rf"<Prefix>({pmcid}\.(\d+))/</Prefix>", resp.text)
    if not versions:
        return None
    folder = max(versions, key=lambda v: int(v[1]))[0]
    return f"{PMC_OA}/{folder}/{folder}.pdf"
