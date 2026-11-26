"""Offline tests for: every sample with its name, no literature values, DOI column, PDF text quirks.

Run:  python -m pytest test_samples.py -q
"""
import csv

import agent
from test_agent import _Call, _fake_chat, _msg   # same scripted fake LLM
from tools import PaperTools

PAPER = ("https://doi.org/10.1016/j.ijhydene.2024.01.306\n"
         "Introduction. Bulk g-C3N4 has a band gap of about 2.7 eV [12], and Wang et al. reported 2.65 eV. "
         "The tangent of the Tauc curves was plotted, so the Eg of 3NBCN and GCN is 2.70 eV and 2.75 eV, respec­ tively. "
         "The SBET of GCN and 3NBCN were 22.5 m2/g and 22.8 m2/g, respectively. "
         "The BET surface area was signiﬁcantly increased.\n"
         "Table 2 Comparison with other photocatalysts\n| TiO2-N | 3.05 eV | [35] |")
BAND_GAP = {"name": "band gap", "parameter": "band gap"}


def _tools():
    return PaperTools(PAPER, BAND_GAP, {})


# ------------------------------------------------------------ all samples, named
def test_main_and_other_samples_are_all_checked_and_saved():
    t = _tools()
    q = "the Eg of 3NBCN and GCN is 2.70 eV and 2.75 eV, respectively."   # model drops the soft hyphen
    out = t.save_result(2.70, unit="eV", sample_name="3NBCN", evidence=q,
                        other_samples=[{"sample_name": "GCN", "value": 2.75, "evidence": q}])
    assert out["saved"] is True and out["n_samples"] == 2
    assert [o["sample_name"] for o in t.result["other_samples"]] == ["GCN"]
    assert t.result["other_samples"][0]["unit"] == "eV"          # inherits the main unit


def test_sample_name_is_required_and_must_be_in_the_paper():
    q = "the Eg of 3NBCN and GCN is 2.70 eV and 2.75 eV, respectively."
    out = _tools().save_result(2.70, unit="eV", evidence=q)
    assert out["saved"] is False and "sample name" in out["problems"][0]
    out = _tools().save_result(2.70, unit="eV", sample_name="CN-opt", evidence=q)
    assert out["saved"] is False and "does not appear in the paper" in out["problems"][0]


def test_wrong_value_for_an_other_sample_rejects_the_save():
    q = "the Eg of 3NBCN and GCN is 2.70 eV and 2.75 eV, respectively."
    out = _tools().save_result(2.70, unit="eV", sample_name="3NBCN", evidence=q,
                               other_samples=[{"sample_name": "GCN", "value": 2.80, "evidence": q}])
    assert out["saved"] is False and any("GCN" in p and "2.8" in p for p in out["problems"])


def test_values_from_other_papers_are_rejected():
    for quote, value, name in [("Bulk g-C3N4 has a band gap of about 2.7 eV [12]", 2.7, "g-C3N4"),
                               ("Wang et al. reported 2.65 eV", 2.65, "g-C3N4"),
                               ("| TiO2-N | 3.05 eV | [35] |", 3.05, "TiO2-N")]:
        out = _tools().save_result(value, unit="eV", sample_name=name, evidence=quote)
        assert out["saved"] is False, quote
        assert any("another paper" in p for p in out["problems"]), quote


def test_no_main_value_promotes_first_other_sample():
    q = "The SBET of GCN and 3NBCN were 22.5 m2/g and 22.8 m2/g, respectively."
    t = PaperTools(PAPER, {"name": "BET surface area", "parameter": "BET surface area"}, {})
    out = t.save_result(None, evidence="", other_samples=[{"sample_name": "GCN", "value": 22.5,
                                                           "unit": "m2/g", "evidence": q}])
    assert out["saved"] is True and t.result["sample_name"] == "GCN" and t.result["other_samples"] == []


def test_ligatures_in_the_paper_do_not_break_the_evidence_check():
    assert _tools().check_evidence("The BET surface area was significantly increased.")["found"] is True


# ------------------------------------------------------------ DOI
def test_find_doi():
    assert agent.find_doi(PAPER) == "10.1016/j.ijhydene.2024.01.306"
    assert agent.find_doi("RSC Adv., 2024, 14, 1 DOI: 10.1039/d4ra04284k. Received") == "10.1039/d4ra04284k"
    assert agent.find_doi("no identifier here") == ""


# ------------------------------------------------------------ whole run: rows, DOI, sample names reused
def test_csv_has_one_row_per_sample_with_doi(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "RESULTS_DIR", str(tmp_path))
    q_eg = "the Eg of 3NBCN and GCN is 2.70 eV and 2.75 eV, respectively."
    q_bet = "The SBET of GCN and 3NBCN were 22.5 m2/g and 22.8 m2/g, respectively."
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "band gap"}, {"name": "BET surface area"}]})),
        _msg(_Call("save_result", {"value": 2.70, "unit": "eV", "sample_name": "3NBCN", "evidence": q_eg,
                                   "other_samples": [{"sample_name": "GCN", "value": 2.75, "evidence": q_eg}]})),
        _msg(_Call("save_result", {"value": 22.8, "unit": "m2/g", "sample_name": "3NBCN", "evidence": q_bet,
                                   "other_samples": [{"sample_name": "GCN", "value": 22.5, "evidence": q_bet}]})),
    ]
    chat = _fake_chat(replies)
    monkeypatch.setattr(agent, "chat", chat)

    final = agent.run_agent(PAPER, "p", "band gap and BET surface area", params={})

    assert final["doi"] == "10.1016/j.ijhydene.2024.01.306"
    assert final["samples"] == ["3NBCN", "GCN"]
    # the BET run was told which sample names were already used
    assert "Sample names already used for this paper" in chat.sent[2][-1]["content"]
    assert "3NBCN, GCN" in chat.sent[2][-1]["content"]

    with open(tmp_path / "p_extraction.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [(r["item"], r["sample_name"], r["is_main"], r["value"], r["doi"]) for r in rows] == [
        ("band gap", "3NBCN", "yes", "2.7", "10.1016/j.ijhydene.2024.01.306"),
        ("band gap", "GCN", "no", "2.75", "10.1016/j.ijhydene.2024.01.306"),
        ("BET surface area", "3NBCN", "yes", "22.8", "10.1016/j.ijhydene.2024.01.306"),
        ("BET surface area", "GCN", "no", "22.5", "10.1016/j.ijhydene.2024.01.306"),
    ]


def test_llm_failure_marks_item_failed_and_keeps_going(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "RESULTS_DIR", str(tmp_path))
    q = "The SBET of GCN and 3NBCN were 22.5 m2/g and 22.8 m2/g, respectively."
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "band gap"}, {"name": "BET surface area"}]})),
        RuntimeError("LLM call failed after retries"),
        _msg(_Call("save_result", {"value": 22.8, "unit": "m2/g", "sample_name": "3NBCN", "evidence": q})),
    ]

    def chat(messages, tools=None):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(agent, "chat", chat)

    final = agent.run_agent(PAPER, "p", "band gap and BET", params={})
    assert [(r["item"], r["status"]) for r in final["results"]] == [("band gap", "failed"),
                                                                    ("BET surface area", "verified")]
