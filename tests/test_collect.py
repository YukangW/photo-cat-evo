"""Offline tests for collect_papers.py / sources.py. A fake HTTP session stands in for OpenAlex, arXiv
and the publisher sites, so no network is used.

Run:  python -m pytest -q
"""
import csv
import json

import pytest
from openpyxl import load_workbook

from PhotoCatEvo import collect_papers as cp
from PhotoCatEvo.sources import SourceError, resolve_journal, search_arxiv, search_openalex

PARAMS = {
    "band gap": {"aliases": ["bandgap", "Eg"]},
    "BET surface area": {"aliases": ["surface area", "BET"]},
    "catalyst mass": {"aliases": ["photocatalyst mass"], "search_papers": False},
}

WORK = {
    "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1039/D4RA04284K",
    "title": "Defective g-C3N4 for photocatalytic H2 evolution", "publication_year": 2024,
    "primary_location": {"source": {"display_name": "RSC Advances", "host_organization_name": "RSC"}},
    "best_oa_location": {"pdf_url": "https://pubs.example.org/d4ra04284k.pdf",
                         "landing_page_url": "https://doi.org/10.1039/d4ra04284k", "license": "cc-by"},
    "open_access": {"oa_status": "gold"},
    "authorships": [{"author": {"display_name": f"Author {i}"}} for i in range(7)],
    "cited_by_count": 12,
    "abstract_inverted_index": {"The": [0], "band": [1], "gap": [2], "narrows": [3]},
}
ARXIV_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1902.10211v1</id>
    <title>Band-gap engineering in perovskite oxysulfides for
      photocatalytic water splitting</title>
    <published>2019-02-26T20:45:41Z</published>
    <summary>We tune the band gap and surface area.</summary>
    <link href="https://arxiv.org/pdf/1902.10211v1" rel="related" type="application/pdf" title="pdf"/>
    <author><name>A. Researcher</name></author>
  </entry>
</feed>"""


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self._payload, self.content = status, payload, content
        self.text = content.decode("utf-8", "ignore") if content else json.dumps(payload or {})

    def json(self):
        return self._payload

    def iter_content(self, size):
        for i in range(0, len(self.content), size):
            yield self.content[i:i + size]


class FakeSession:
    """Answers by URL. Records every request so tests can check what was asked."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, **kw):
        self.calls.append((url, params or {}))
        for key, resp in self.routes.items():
            if key in url:
                return resp(params or {}) if callable(resp) else resp
        return FakeResponse(404, content=b"not found")


def _sources(params):
    results = {"RSC Advances": [{"id": "https://openalex.org/S1", "display_name": "RSC Advances",
                                 "issn_l": "2046-2069", "is_oa": True, "is_in_doaj": True, "type": "journal"}],
               "Closed Journal": [{"id": "https://openalex.org/S2", "display_name": "Closed Journal",
                                   "is_oa": False, "type": "journal"}]}
    return FakeResponse(payload={"results": results.get(params["search"], [])})


ROUTES = {
    "api.openalex.org/sources": _sources,
    "api.openalex.org/works": FakeResponse(payload={"results": [WORK]}),
    "export.arxiv.org": FakeResponse(content=ARXIV_XML),
    "pubs.example.org": FakeResponse(content=b"%PDF-1.7 fake pdf"),
    "arxiv.org/pdf": FakeResponse(content=b"<html>blocked</html>"),
}


# ------------------------------------------------------------ sources
def test_openalex_record_fields():
    session = FakeSession(ROUTES)
    r = search_openalex("q", ["S1"], 5, 2022, session)[0]
    assert (r["doi"], r["journal"], r["license"], r["year"]) == ("10.1039/d4ra04284k", "RSC Advances", "cc-by", 2024)
    assert r["authors"].endswith("; et al.") and r["abstract"] == "The band gap narrows"
    flt = session.calls[0][1]["filter"]
    assert "is_oa:true" in flt and "primary_location.source.id:S1" in flt and "from_publication_date:2022" in flt


def test_arxiv_record_fields():
    r = search_arxiv("q", 5, FakeSession(ROUTES))[0]
    assert r["arxiv_id"] == "1902.10211v1" and r["pdf_url"] == "https://arxiv.org/pdf/1902.10211v1"
    assert r["title"].startswith("Band-gap engineering") and "\n" not in r["title"] and r["year"] == 2019


def test_resolve_journal():
    j = resolve_journal("RSC Advances", FakeSession(ROUTES))
    assert (j["id"], j["is_oa"], j["exact_match"]) == ("S1", True, True)
    assert resolve_journal("Nothing", FakeSession(ROUTES)) is None


def test_openalex_budget_message():
    with pytest.raises(SourceError, match="OPENALEX_API_KEY"):
        search_openalex("q", session=FakeSession({"openalex": FakeResponse(429)}))


def test_no_personal_details_sent(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    session = FakeSession(ROUTES)
    search_openalex("q", session=session)
    assert "mailto" not in session.calls[0][1] and "api_key" not in session.calls[0][1]


# ------------------------------------------------------------ journals
def test_resolve_journals_skips_closed_and_missing(tmp_path):
    found, skipped = cp.resolve_journals(["RSC Advances", "Closed Journal", "Nothing"],
                                         str(tmp_path / "cache.json"), FakeSession(ROUTES))
    assert [j["id"] for j in found] == ["S1"]
    assert any("not list" in s for s in skipped) and any("not found" in s for s in skipped)


def test_journal_lookups_are_cached(tmp_path):
    cache = str(tmp_path / "cache.json")
    cp.resolve_journals(["RSC Advances"], cache, FakeSession(ROUTES))
    second = FakeSession(ROUTES)
    cp.resolve_journals(["RSC Advances"], cache, second)
    assert second.calls == []


def test_shipped_journal_list_loads():
    names = cp.load_journal_names()
    assert "RSC Advances" in names and len(names) == len(set(names))


# ------------------------------------------------------------ queries from the dictionary
def test_build_queries_from_dictionary():
    qs = cp.build_queries(PARAMS)
    assert [q["parameter"] for q in qs] == ["band gap", "BET surface area"]      # catalyst mass skipped
    assert qs[0]["openalex"] == '(photocatalytic OR photocatalysis OR photocatalyst) AND ("band gap" OR "bandgap")'
    assert 'all:"BET"' in qs[1]["arxiv"] and '"Eg"' not in qs[0]["openalex"]   # short ambiguous alias dropped


def test_build_queries_named_and_unknown():
    assert [q["parameter"] for q in cp.build_queries(PARAMS, ["catalyst mass"])] == ["catalyst mass"]
    with pytest.raises(ValueError, match="Not in the dictionary"):
        cp.build_queries(PARAMS, ["zeta potential"])


def test_parameters_mentioned():
    r = {"title": "Tuning the bandgap of TiO2", "abstract": "High BET surface area. Begging."}
    assert cp.parameters_mentioned(r, PARAMS) == ["band gap", "BET surface area"]


# ------------------------------------------------------------ merge
def test_merge_dedupes_and_keeps_download_status():
    old = [{"doi": "10.1/x", "title": "A", "matched_parameters": "band gap", "download_status": "downloaded",
            "pdf_file": "papers/collected/10.1_x.pdf"}]
    new = [{"doi": "10.1/X", "title": "A", "matched_parameters": "BET surface area", "download_status": ""},
           {"arxiv_id": "1902.10211v2", "title": "B"}, {"arxiv_id": "1902.10211v1", "title": "B"}]
    out = cp.merge(old, new)
    assert len(out) == 2
    assert out[0]["download_status"] == "downloaded"
    assert out[0]["matched_parameters"] == "band gap; BET surface area"


# ------------------------------------------------------------ download
def test_download_pdf(tmp_path):
    session = FakeSession(ROUTES)
    rec = {"doi": "10.1039/d4ra04284k", "pdf_url": "https://pubs.example.org/d4ra04284k.pdf"}
    path, status, source = cp.download_pdf(rec, str(tmp_path), session)
    assert status == "downloaded" and path.endswith("10.1039_d4ra04284k.pdf") and source == "pubs.example.org"
    assert open(path, "rb").read().startswith(b"%PDF")
    assert cp.download_pdf(rec, str(tmp_path), session)[1] == "already_downloaded"


@pytest.mark.parametrize("rec, status", [
    ({"arxiv_id": "1", "pdf_url": "https://arxiv.org/pdf/1"}, "failed: arxiv.org not_pdf"),   # web page
    ({"doi": "10.1/y", "pdf_url": "https://nowhere.example/y.pdf"}, "failed: nowhere.example http_404"),
    ({"doi": "10.1/z", "pdf_url": ""}, "no_pdf_link"),
])
def test_download_failures_are_reported(tmp_path, rec, status):
    path, got, _ = cp.download_pdf(rec, str(tmp_path), FakeSession(ROUTES))
    assert path == "" and got.startswith(status)
    assert not list(tmp_path.glob("*.pdf"))


# ------------------------------------------------------------ end to end (offline)
def test_main_end_to_end(tmp_path, monkeypatch):
    session = FakeSession(ROUTES)
    from PhotoCatEvo import sources
    monkeypatch.setattr(sources, "_session", session)
    monkeypatch.setattr(cp.requests, "Session", lambda: session)
    monkeypatch.setattr(cp.time, "sleep", lambda s: None)
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    monkeypatch.setattr(cp, "load_parameters", lambda: PARAMS)
    monkeypatch.setattr(cp, "CACHE_FILE", str(tmp_path / "cache.json"))
    journals = tmp_path / "j.json"
    journals.write_text(json.dumps({"journals": ["RSC Advances", "Closed Journal"]}), encoding="utf-8")
    out, pdfs = str(tmp_path / "res" / "papers"), str(tmp_path / "pdfs")

    cp.main(["--journals", str(journals), "--out", out, "--pdf-dir", pdfs, "--per-query", "5"])

    with open(out + ".csv", encoding="utf-8-sig") as f:
        rows = {r["title"][:20]: r for r in csv.DictReader(f)}
    assert len(rows) == 2                                    # same paper from 2 queries merged
    oa = rows["Defective g-C3N4 for"]
    assert oa["matched_parameters"] == "band gap; BET surface area"
    assert oa["download_status"] == "downloaded" and oa["parameters_mentioned"] == "band gap"
    assert rows["Band-gap engineering"]["download_status"].startswith("failed: arxiv.org not_pdf")
    assert load_workbook(out + ".xlsx").active.max_row == 3

    # the works search was limited to the open-access journal only
    works_filters = [p["filter"] for u, p in session.calls if "openalex.org/works" in u]
    assert all("primary_location.source.id:S1" in f and "S2" not in f for f in works_filters)

    # a second run keeps the earlier download and does not fetch it again
    session.calls.clear()
    cp.main(["--journals", str(journals), "--out", out, "--pdf-dir", pdfs, "--per-query", "5"])
    assert not any("pubs.example.org" in u for u, _ in session.calls)


# ------------------------------------------------------------ PubMed Central open-access copy
PMC_LISTING = FakeResponse(content=b"<ListBucketResult><CommonPrefixes><Prefix>PMC123.1/</Prefix></CommonPrefixes>"
                                   b"<CommonPrefixes><Prefix>PMC123.2/</Prefix></CommonPrefixes></ListBucketResult>")
PMC_ROUTES = {
    "pmc-oa-opendata.s3.amazonaws.com/PMC123.2/": FakeResponse(content=b"%PDF-1.5 from pmc"),
    "pmc-oa-opendata.s3.amazonaws.com": PMC_LISTING,
    "blocked.example": FakeResponse(403, content=b"<html>Just a moment...</html>"),
    "repository.example": FakeResponse(content=b"%PDF-1.4 repository copy"),
}


def test_pmc_pdf_url_picks_newest_version():
    from PhotoCatEvo.sources import pmc_pdf_url
    assert pmc_pdf_url("PMC123", FakeSession(PMC_ROUTES)).endswith("/PMC123.2/PMC123.2.pdf")
    assert pmc_pdf_url("not-an-id", FakeSession(PMC_ROUTES)) is None


def test_download_prefers_pmc_copy(tmp_path):
    rec = {"doi": "10.1/a", "pmcid": "PMC123", "pdf_url": "https://blocked.example/a.pdf"}
    path, status, source = cp.download_pdf(rec, str(tmp_path), FakeSession(PMC_ROUTES))
    assert status == "downloaded" and source == "PubMed Central (open-access data)"
    assert open(path, "rb").read() == b"%PDF-1.5 from pmc"


def test_download_falls_back_to_other_copies(tmp_path):
    rec = {"doi": "10.1/b", "pdf_url": "https://blocked.example/b.pdf",
           "other_pdf_urls": "https://repository.example/b.pdf"}
    path, status, source = cp.download_pdf(rec, str(tmp_path), FakeSession(PMC_ROUTES))
    assert (status, source) == ("downloaded", "repository.example")


def test_download_all_copies_refused(tmp_path):
    rec = {"doi": "10.1/c", "pdf_url": "https://blocked.example/c.pdf"}
    _, status, _ = cp.download_pdf(rec, str(tmp_path), FakeSession(PMC_ROUTES))
    assert status.startswith("failed: blocked.example http_403") and "open landing_url by hand" in status


def test_openalex_record_collects_all_open_copies():
    work = {**WORK, "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/9696975"},
            "locations": [{"pdf_url": "https://pubs.example.org/d4ra04284k.pdf", "is_oa": True},
                          {"pdf_url": "https://repository.example/x.pdf", "is_oa": True},
                          {"pdf_url": "https://paywalled.example/x.pdf", "is_oa": False}]}
    session = FakeSession({"openalex.org/works": FakeResponse(payload={"results": [work]})})
    r = search_openalex("q", session=session)[0]
    assert r["pmcid"] == "PMC9696975"
    assert r["other_pdf_urls"] == "https://repository.example/x.pdf"            # closed copy left out


def test_arxiv_year_filter_is_in_the_query():
    session = FakeSession(ROUTES)
    search_arxiv('all:"band gap"', 5, session, from_year=2022)
    assert "submittedDate:[202201010000 TO" in session.calls[0][1]["search_query"]
