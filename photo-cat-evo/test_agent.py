"""Offline tests: the LLM is replaced by a scripted fake, so no API key or credit is needed.

Run:  python -m pytest test_agent.py -q
"""
import json
from types import SimpleNamespace

import pytest

import agent
from tools import PaperTools, PlanTools

PAPER = ("The optimised sample CN-550 showed a band gap of 2.65 eV from the Tauc plot. "
         "Its BET surface area was 45.3 m2 g-1, higher than bulk g-C3N4 (10.2 m2 g-1). "
         "CN-550 gave an H2 evolution rate of 1250 umol g-1 h-1 under visible light.")


# ------------------------------------------------------------ fake LLM
class _Call:
    _n = 0

    def __init__(self, name, args):
        _Call._n += 1
        self.id = f"call_{_Call._n}"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))

    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


def _msg(*calls, content=""):
    return SimpleNamespace(content=content, tool_calls=list(calls) or None, model_extra={})


def _fake_chat(replies):
    """Return a chat() that hands out the scripted replies in order and records what it was sent."""
    sent = []

    def chat(messages, tools=None):
        sent.append([dict(m) for m in messages])
        return replies.pop(0)

    chat.sent = sent
    return chat


@pytest.fixture(autouse=True)
def _tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "RESULTS_DIR", str(tmp_path))


# ------------------------------------------------------------ tests
def test_multi_item_prompt_runs_once_per_item_in_order(monkeypatch):
    targets = [{"name": "band gap", "unit": "eV"},
               {"name": "BET surface area", "unit": "m2 g-1"},
               {"name": "H2 evolution rate", "unit": "umol g-1 h-1"}]
    replies = [
        _msg(_Call("set_targets", {"targets": targets})),
        # item 1: search, then save
        _msg(_Call("search_text", {"keyword": "band gap"})),
        _msg(_Call("save_result", {"value": 2.65, "unit": "eV", "sample_name": "CN-550",
                                   "evidence": "CN-550 showed a band gap of 2.65 eV"})),
        # item 2: save straight away
        _msg(_Call("save_result", {"value": 45.3, "unit": "m2 g-1", "sample_name": "CN-550",
                                   "evidence": "Its BET surface area was 45.3 m2 g-1"})),
        # item 3: save
        _msg(_Call("save_result", {"value": 1250, "unit": "umol g-1 h-1", "sample_name": "CN-550",
                                   "evidence": "H2 evolution rate of 1250 umol g-1 h-1"})),
    ]
    chat = _fake_chat(replies)
    monkeypatch.setattr(agent, "chat", chat)

    final = agent.run_agent(PAPER, "paper", "Find the band gap, BET surface area and H2 rate")

    assert [r["item"] for r in final["results"]] == ["band gap", "BET surface area", "H2 evolution rate"]
    assert [r["value"] for r in final["results"]] == [2.65, 45.3, 1250]
    assert all(r["status"] == "verified" for r in final["results"])
    assert replies == []          # every scripted reply was used, no extra LLM calls

    # The run for item 2 starts a FRESH conversation that names item 2 and carries item 1's result.
    item2_task = chat.sent[3][-1]["content"]
    assert "Extract this data item now: BET surface area" in item2_task
    assert "band gap: 2.65 eV for sample CN-550" in item2_task


def test_rejected_result_is_retried_before_moving_on(monkeypatch):
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "band gap", "unit": "eV"}]})),
        _msg(_Call("save_result", {"value": 2.9, "evidence": "CN-550 showed a band gap of 2.65 eV"})),
        _msg(_Call("save_result", {"value": 2.65, "evidence": "CN-550 showed a band gap of 2.65 eV"})),
    ]
    monkeypatch.setattr(agent, "chat", _fake_chat(replies))

    final = agent.run_agent(PAPER, "paper", "band gap")

    assert final["results"][0]["value"] == 2.65
    assert final["results"][0]["steps"] == 2


def test_planner_fallback_uses_whole_prompt(monkeypatch):
    replies = [_msg(content="no tools"), _msg(content="no"), _msg(content="still no"),
               _msg(_Call("save_result", {"value": None, "evidence": ""}))]
    monkeypatch.setattr(agent, "chat", _fake_chat(replies))

    final = agent.run_agent(PAPER, "paper", "Find the quantum yield")

    assert final["results"][0]["item"] == "Find the quantum yield"
    assert final["results"][0]["status"] == "not_reported"


def test_set_targets_cleans_and_caps():
    plan = PlanTools(max_targets=2)
    assert plan.set_targets([])["saved"] is False
    plan.set_targets(["band gap", {"name": " BET "}, {"name": ""}, {"name": "rate"}])
    assert [t["name"] for t in plan.targets] == ["band gap", "BET"]


def test_save_result_checks():
    t = PaperTools(PAPER, {"name": "band gap", "unit": "eV"})
    assert t.save_result(2.65, evidence="not in the paper at all")["saved"] is False
    assert t.save_result(9.0, evidence="band gap of 2.65 eV")["saved"] is False      # value not in quote
    assert t.save_result("2.65", evidence="band gap of 2.65 eV")["saved"] is True     # string -> number


# ------------------------------------------------------------ value / range / error columns
PAPER2 = ("Both samples show high specific surface areas of >160 m2 g−1. "
          "The band gap of DCN is 2.72 ± 0.02 eV, within the 2.6–2.8 eV reported for g-C3N4. "
          "The H2 evolution rate of DCN was 6.0 mmol g−1 h−1 (error 0.3, standard deviation of three runs).")


def test_range_only_when_no_specific_value():
    t = PaperTools(PAPER2, {"name": "BET surface area"})
    out = t.save_result(None, unit="m2 g-1", range=">160",
                        evidence="high specific surface areas of >160 m2 g−1")
    assert out["status"] == "range_only"
    assert t.result["value"] is None and t.result["range"] == ">160"


def test_value_with_error_and_dash_range():
    t = PaperTools(PAPER2, {"name": "band gap"})
    out = t.save_result(2.72, unit="eV", error="± 0.02", range="2.6-2.8",
                        evidence="The band gap of DCN is 2.72 ± 0.02 eV, within the 2.6–2.8 eV")
    assert out["status"] == "verified"
    assert (t.result["value"], t.result["error"], t.result["range"]) == (2.72, "± 0.02", "2.6-2.8")


def test_error_from_separate_quote():
    t = PaperTools(PAPER2, {"name": "H2 evolution rate"})
    out = t.save_result(6.0, evidence="H2 evolution rate of DCN was 6.0 mmol",
                        error="0.3", error_evidence="(error 0.3, standard deviation of three runs)")
    assert out["status"] == "verified"


def test_made_up_range_or_error_is_rejected():
    t = PaperTools(PAPER2, {"name": "band gap"})
    assert t.save_result(2.72, evidence="band gap of DCN is 2.72 ± 0.02 eV", error="± 0.05")["saved"] is False
    assert t.save_result(None, range="2.0-2.4", evidence="within the 2.6–2.8 eV")["saved"] is False
    assert t.result is None


def test_not_reported_keeps_quote_and_note():
    t = PaperTools(PAPER2, {"name": "quantum yield"})
    t.save_result(None, evidence="", note="Only in Fig. 5")
    assert t.result["status"] == "not_reported" and t.result["note"] == "Only in Fig. 5"


def test_csv_has_value_range_error_columns(monkeypatch, tmp_path):
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "band gap"}, {"name": "BET surface area"}]})),
        _msg(_Call("save_result", {"value": 2.72, "unit": "eV", "error": "± 0.02",
                                   "evidence": "The band gap of DCN is 2.72 ± 0.02 eV"})),
        _msg(_Call("save_result", {"value": None, "range": ">160", "unit": "m2 g-1",
                                   "evidence": "high specific surface areas of >160 m2 g−1"})),
    ]
    monkeypatch.setattr(agent, "chat", _fake_chat(replies))
    agent.run_agent(PAPER2, "p2", "band gap and BET")

    import csv
    with open(tmp_path / "p2_extraction.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [(r["item"], r["value"], r["range"], r["error"], r["status"]) for r in rows] == [
        ("band gap", "2.72", "", "± 0.02", "verified"),
        ("BET surface area", "", ">160", "", "range_only"),
    ]
