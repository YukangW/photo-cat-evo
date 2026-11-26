"""Offline tests: the LLM is replaced by a scripted fake, so no API key or credit is needed.

Run:  python -m pytest test_agent.py -q
"""
import json
from types import SimpleNamespace

import httpx
import pytest

from PhotoCatEvo import agent, llm
from PhotoCatEvo.tools import PaperTools, PlanTools

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

    def chat(messages, tools=None, *, token_usage=None):
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


def test_extraction_metrics_in_cli_and_trace(monkeypatch, tmp_path, capsys):
    targets = [{"name": name} for name in ("BET surface area", "band gap", "quantum yield", "lifetime")]
    replies = iter([
        (_msg(_Call("set_targets", {"targets": targets})), 9000, 900),
        (_msg(_Call("search_text", {"keyword": "BET"})), 100, 10),
        llm.APIConnectionError(request=httpx.Request("POST", "https://example.test")),
        (_msg(_Call("save_result", {"value": 45.3, "unit": "m2 g-1", "sample_name": "CN-550",
                                   "evidence": "Its BET surface area was 45.3 m2 g-1",
                                   "other_samples": [{"sample_name": "bulk g-C3N4", "value": 10.2,
                                                      "evidence": "bulk g-C3N4 (10.2 m2 g-1)"}]})), 200, 20),
        (_msg(_Call("save_result", {"value": 2.65, "unit": "eV", "sample_name": "CN-550",
                                   "evidence": "CN-550 showed a band gap of 2.65 eV"})), 300, 30),
        (_msg(_Call("save_result", {"value": None, "sample_name": "unverified name"})), None, None),
        (None, 400, 40),  # usage still counts when the response has no choices
    ])

    def create(**kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        message, prompt, completion = reply
        usage = (None if prompt is None else
                 SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)] if message else [], usage=usage)

    sleeps = []
    clock = iter([10, 11, 16, 17, 20, 21, 22, 23, 25, 26])
    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    monkeypatch.setattr(llm, "time", SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(agent, "perf_counter", lambda: next(clock))
    paper_path = tmp_path / "paper.txt"
    paper_path.write_text(PAPER, encoding="utf-8")

    agent.main([str(paper_path), "Extract BET surface area, band gap, quantum yield and lifetime"])

    assert sleeps == [5]
    trace = json.loads((tmp_path / "paper_extraction_trace.json").read_text(encoding="utf-8"))
    assert trace["metrics"] == {
        "elapsed_seconds": 16, "parameters_attempted": 4, "parameters_extracted": 2, "sample_count": 2,
        "token_usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
                        "requests_with_usage": 4, "requests_without_usage": 2},
    }
    assert [item["item"] for item in trace["items"]] == [target["name"] for target in targets]
    assert [(item["metrics"]["status"], item["metrics"]["elapsed_seconds"], item["metrics"]["token_usage"])
            for item in trace["items"]] == [
        ("verified", 5, {"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 330,
                         "requests_with_usage": 2, "requests_without_usage": 1}),
        ("verified", 3, {"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 330,
                         "requests_with_usage": 1, "requests_without_usage": 0}),
        ("not_reported", 1, {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                             "requests_with_usage": 0, "requests_without_usage": 1}),
        ("failed", 2, {"prompt_tokens": 400, "completion_tokens": 40, "total_tokens": 440,
                       "requests_with_usage": 1, "requests_without_usage": 0}),
    ]
    output = capsys.readouterr().out
    assert "4 parameters attempted, 2 extracted, 2 samples, 16.00 s" in output
    assert "1,000 input, 100 output, 1,100 total tokens (partial, 4/6" in output
    assert "BET surface area [verified]: 5.00 s, 300 input, 30 output, 330 total tokens" in output
    assert "quantum yield [not_reported]: 1.00 s, token usage unavailable" in output


def test_rejected_result_is_retried_before_moving_on(monkeypatch):
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "band gap", "unit": "eV"}]})),
        _msg(_Call("save_result", {"value": 2.9, "sample_name": "CN-550", "evidence": "CN-550 showed a band gap of 2.65 eV"})),
        _msg(_Call("save_result", {"value": 2.65, "sample_name": "CN-550", "evidence": "CN-550 showed a band gap of 2.65 eV"})),
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
    assert t.save_result(2.65, sample_name="CN-550", evidence="not in the paper at all")["saved"] is False
    assert t.save_result(9.0, sample_name="CN-550", evidence="band gap of 2.65 eV")["saved"] is False      # value not in quote
    assert t.save_result("2.65", sample_name="CN-550", evidence="band gap of 2.65 eV")["saved"] is True     # string -> number


# ------------------------------------------------------------ value / range / error columns
PAPER2 = ("Both samples show high specific surface areas of >160 m2 g−1. "
          "The band gap of DCN is 2.72 ± 0.02 eV, within the 2.6–2.8 eV reported for g-C3N4. "
          "The H2 evolution rate of DCN was 6.0 mmol g−1 h−1 (error 0.3, standard deviation of three runs).")


def test_range_only_when_no_specific_value():
    t = PaperTools(PAPER2, {"name": "BET surface area"})
    out = t.save_result(None, unit="m2 g-1", range=">160", sample_name="DCN",
                        evidence="high specific surface areas of >160 m2 g−1")
    assert out["status"] == "range_only"
    assert t.result["value"] is None and t.result["range"] == ">160"


def test_value_with_error_and_dash_range():
    t = PaperTools(PAPER2, {"name": "band gap"})
    out = t.save_result(2.72, unit="eV", error="± 0.02", range="2.6-2.8", sample_name="DCN",
                        evidence="The band gap of DCN is 2.72 ± 0.02 eV, within the 2.6–2.8 eV")
    assert out["status"] == "verified"
    assert (t.result["value"], t.result["error"], t.result["range"]) == (2.72, "± 0.02", "2.6-2.8")


def test_error_from_separate_quote():
    t = PaperTools(PAPER2, {"name": "H2 evolution rate"})
    out = t.save_result(6.0, sample_name="DCN", evidence="H2 evolution rate of DCN was 6.0 mmol",
                        error="0.3", error_evidence="(error 0.3, standard deviation of three runs)")
    assert out["status"] == "verified"


def test_made_up_range_or_error_is_rejected():
    t = PaperTools(PAPER2, {"name": "band gap"})
    assert t.save_result(2.72, sample_name="DCN", evidence="band gap of DCN is 2.72 ± 0.02 eV", error="± 0.05")["saved"] is False
    assert t.save_result(None, range="2.0-2.4", evidence="within the 2.6–2.8 eV")["saved"] is False
    assert t.result is None


def test_not_reported_keeps_quote_and_note():
    t = PaperTools(PAPER2, {"name": "quantum yield"})
    t.save_result(None, evidence="", note="Only in Fig. 5")
    assert t.result["status"] == "not_reported" and t.result["note"] == "Only in Fig. 5"


def test_csv_has_value_range_error_columns(monkeypatch, tmp_path):
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "band gap"}, {"name": "BET surface area"}]})),
        _msg(_Call("save_result", {"value": 2.72, "unit": "eV", "error": "± 0.02", "sample_name": "DCN",
                                   "evidence": "The band gap of DCN is 2.72 ± 0.02 eV"})),
        _msg(_Call("save_result", {"value": None, "range": ">160", "unit": "m2 g-1", "sample_name": "DCN",
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
