"""Offline tests for the parameter dictionary and how the agent uses it.

Run:  python -m pytest -q
"""
import json

import pytest

from PhotoCatEvo import agent, parameters
from PhotoCatEvo.parameters import add_parameter, find_parameter, load_parameters, units_match
from PhotoCatEvo.tools import PaperTools
from test_agent import _Call, _fake_chat, _msg  # noqa: F401  (also brings in the results-dir fixture)
from test_agent import _tmp_results  # noqa: F401

PARAMS = {
    "band gap": {"aliases": ["bandgap", "Eg"], "unit": "eV", "plausible_range": [1.0, 4.5],
                 "search_terms": ["Tauc"], "description": "Optical band gap."},
    "BET surface area": {"aliases": ["surface area", "SSA"], "unit": "m2 g-1", "plausible_range": [0.1, 3000]},
    "H2 evolution rate": {"aliases": ["hydrogen evolution rate"], "unit": "µmol g-1 h-1",
                          "plausible_range": [0.01, 200000]},
}


# ------------------------------------------------------------ the shipped file
def test_shipped_dictionary_is_valid():
    params = load_parameters()
    assert {"band gap", "BET surface area", "H2 evolution rate"} <= params.keys()
    for name, entry in params.items():
        assert set(entry) <= set(parameters.FIELDS), name
        if entry.get("plausible_range"):
            lo, hi = entry["plausible_range"]
            assert lo <= hi, name


# ------------------------------------------------------------ lookup
@pytest.mark.parametrize("item, expected", [
    ("band gap", "band gap"),
    ("Bandgap", "band gap"),                                  # alias, any case
    ("Eg", "band gap"),                                       # short alias, exact match only
    ("specific surface area", "BET surface area"),            # not an alias, but contains one
    ("H2 evolution rate under visible light", "H2 evolution rate"),
    ("hydrogen-evolution rate", "H2 evolution rate"),         # punctuation ignored
    ("quantum yield", None),
    ("leg length", None),                                     # 'Eg' must not match inside a word
])
def test_find_parameter(item, expected):
    match = find_parameter(item, PARAMS)
    assert (match[0] if match else None) == expected


@pytest.mark.parametrize("a, b, same", [
    ("m2 g-1", "m2/g", True),
    ("m2 g−1", "m²·g⁻¹", True),
    ("µmol g-1 h-1", "umol/h/g", True),
    ("µmol g-1 h-1", "μmol h−1 g−1", True),
    ("µmol g-1 h-1", "mmol g-1 h-1", False),
    ("eV", "eV", True),
    ("eV", "", False),
])
def test_units_match(a, b, same):
    assert units_match(a, b) is same


# ------------------------------------------------------------ adding entries
def test_add_parameter_saves_and_refuses_duplicates(tmp_path):
    path = str(tmp_path / "p.json")
    add_parameter("zeta potential", {"unit": "mV", "plausible_range": [-100, 100], "aliases": ["ζ-potential"],
                                     "description": ""}, path)
    saved = json.load(open(path, encoding="utf-8"))
    assert saved == {"zeta potential": {"unit": "mV", "plausible_range": [-100, 100],
                                        "aliases": ["ζ-potential"]}}      # empty fields dropped
    with pytest.raises(ValueError, match="already in the dictionary"):
        add_parameter("Zeta Potential", {"unit": "mV"}, path)
    add_parameter("zeta potential", {"unit": "V"}, path, overwrite=True)
    assert load_parameters(path)["zeta potential"] == {"unit": "V"}


def test_add_parameter_rejects_bad_range(tmp_path):
    with pytest.raises(ValueError, match="min <= max"):
        add_parameter("x", {"plausible_range": [5, 1]}, str(tmp_path / "p.json"))


def test_cli_add_list_show(tmp_path, capsys):
    path = str(tmp_path / "p.json")
    parameters.main(["--file", path, "add", "crystallinity", "--unit", "%", "--range", "0", "100",
                     "--aliases", "degree of crystallinity, crystallinity index", "--search", "XRD,crystallinity"])
    parameters.main(["--file", path, "list"])
    parameters.main(["--file", path, "show", "crystallinity index"])
    out = capsys.readouterr().out
    assert "Added 'crystallinity'" in out
    assert "1 parameter(s)" in out
    assert "Expected range: 0.0-100.0 %" in out


def test_missing_or_broken_file(tmp_path):
    assert load_parameters(str(tmp_path / "none.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text('{"band gap": {"unit": "eV",}}', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_parameters(str(bad))


# ------------------------------------------------------------ checks use the dictionary
PAPER = ("The band gap of CN-550 is 7.5 eV. Its surface area was 45.3 m2/g. "
         "The H2 evolution rate was 6.0 mmol g−1 h−1, and 6000 µmol g−1 h−1 in the second run.")


def test_range_from_dictionary_rejects_implausible_value():
    t = PaperTools(PAPER, {"name": "band gap", "parameter": "band gap"}, PARAMS)
    out = t.save_result(7.5, unit="eV", sample_name="CN-550", evidence="The band gap of CN-550 is 7.5 eV")
    assert out["saved"] is False and "outside the expected range" in out["problems"][0]


def test_range_check_converts_other_units_with_pint():
    t = PaperTools(PAPER, {"name": "H2 evolution rate", "parameter": "H2 evolution rate"}, PARAMS)
    ok = t.check_plausible(6.0, "mmol g−1 h−1")
    assert ok["plausible"] is True and "(= 6000.0 µmol g-1 h-1)" in ok["reason"]
    assert t.check_plausible(6000, "µmol/g/h")["plausible"] is True
    assert t.check_plausible(900, "mmol g−1 h−1")["plausible"] is False      # = 900000 µmol, too high
    assert "not checked" in t.check_plausible(6.0, "a.u.")["reason"]          # unreadable unit -> skipped


def test_item_not_in_dictionary_has_no_range():
    t = PaperTools(PAPER, {"name": "quantum yield", "parameter": ""}, PARAMS)
    assert t.range is None


# ------------------------------------------------------------ the agent calls the dictionary
def test_agent_sends_dictionary_to_planner_and_notes_to_extractor(monkeypatch):
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "bandgap"}, {"name": "quantum yield"}]})),
        _msg(_Call("save_result", {"value": None, "evidence": ""})),
        _msg(_Call("save_result", {"value": None, "evidence": ""})),
    ]
    chat = _fake_chat(replies)
    monkeypatch.setattr(agent, "chat", chat)

    final = agent.run_agent(PAPER, "p", "Find the bandgap and quantum yield", params=PARAMS)

    planner_system = chat.sent[0][0]["content"]
    assert "Parameter dictionary:" in planner_system and "- band gap [eV] (also called: bandgap, Eg)" in planner_system

    item1_task = chat.sent[1][-1]["content"]
    assert "Notes on 'band gap' from the parameter dictionary" in item1_task
    assert "Good search terms: Tauc" in item1_task and "Expected range: 1.0-4.5 eV" in item1_task

    item2_task = chat.sent[2][-1]["content"]
    assert "not in the parameter dictionary. Still extract it" in item2_task

    assert [r["parameter"] for r in final["results"]] == ["band gap", ""]


def test_item_not_in_dictionary_is_still_extracted_and_verified(monkeypatch):
    paper = "The average fluorescence lifetime of DCN is 8.21 ns, longer than CN (5.4 ns)."
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "fluorescence lifetime", "unit": "ns"}]})),
        _msg(_Call("search_text", {"keyword": "lifetime"})),
        _msg(_Call("save_result", {"value": 8.21, "unit": "ns", "sample_name": "DCN",
                                   "evidence": "The average fluorescence lifetime of DCN is 8.21 ns"})),
    ]
    monkeypatch.setattr(agent, "chat", _fake_chat(replies))

    final = agent.run_agent(paper, "p", "Find the fluorescence lifetime", params=PARAMS)

    r = final["results"][0]
    assert (r["parameter"], r["value"], r["status"]) == ("", 8.21, "verified")


def test_item_not_in_dictionary_still_needs_real_evidence(monkeypatch):
    t = PaperTools("The lifetime of DCN is 8.21 ns.", {"name": "fluorescence lifetime", "parameter": ""}, PARAMS)
    assert t.save_result(9.9, sample_name="DCN", evidence="The lifetime of DCN is 8.21 ns")["saved"] is False
    assert t.save_result(8.21, sample_name="DCN", evidence="The lifetime of DCN is 8.21 ns")["saved"] is True
