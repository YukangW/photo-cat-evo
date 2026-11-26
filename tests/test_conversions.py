"""Offline tests for the advanced conversions: Pint contexts (eV <-> nm), reference electrodes + pH,
catalyst-mass normalisation, molar mass, and the agent extracting the extra values it needs.

Run:  python -m pytest -q
"""
import csv

import pytest

from PhotoCatEvo import agent
from PhotoCatEvo.conversions import (NeedsValue, amount_rate_converter, convert_potential, find_reference, molar_mass,
                         potential_shift, split_amount)
from PhotoCatEvo.parameters import add_parameter
from PhotoCatEvo.standardise import standardise
from test_agent import _Call, _fake_chat, _msg, _tmp_results  # noqa: F401  (fixture: results go to tmp)
from PhotoCatEvo.tools import PaperTools
from PhotoCatEvo.units import UnitError, convert, convert_text, needs_context

SP = ["spectroscopy"]


# ------------------------------------------------------------ step 1: Pint contexts
@pytest.mark.parametrize("value, frm, to, ctx, expected", [
    (2.72, "eV", "nm", SP, 455.82),
    (420, "nm", "eV", SP, 2.952),
    (2.72, "eV", "cm-1", SP, 21938.3),
    (2.72, "eV", "THz", SP, 657.69),
    (0.025, "eV", "K", ["boltzmann"], 290.11),
])
def test_contexts(value, frm, to, ctx, expected):
    assert convert(value, frm, to, ctx) == pytest.approx(expected, rel=1e-4)


def test_contexts_are_opt_in():
    with pytest.raises(UnitError, match="measure different things"):
        convert(2.72, "eV", "nm")
    with pytest.raises(UnitError, match="Unknown context"):
        convert(2.72, "eV", "nm", ["magic"])


def test_needs_context():
    assert needs_context("nm", "eV", ["boltzmann", "spectroscopy"]) == "spectroscopy"
    assert needs_context("mmol", "µmol", SP) is None


def test_range_through_inverse_conversion_flips_order_and_signs():
    assert convert_text("2.6-2.8", "eV", "nm", contexts=SP) == "442.801-476.862"
    assert convert_text(">400", "nm", "eV", contexts=SP) == "<3.0996"


def test_error_through_inverse_conversion_needs_and_uses_value():
    assert convert_text("± 0.02", "eV", "nm", as_error=True, contexts=SP, at=2.72) == "± 3.35183"
    with pytest.raises(UnitError, match="without its value"):
        convert_text("± 0.02", "eV", "nm", as_error=True, contexts=SP)


# ------------------------------------------------------------ step 2a: reference electrodes
@pytest.mark.parametrize("text, ref", [
    ("V vs. RHE", "RHE"), ("-0.4 V vs Ag/AgCl", "Ag/AgCl"), ("vs. Ag/AgCl (3 M KCl)", "Ag/AgCl (3 M KCl)"),
    ("V vs. SCE", "SCE"), ("vs. the NHE", "NHE"), ("1.2 V", None),
])
def test_find_reference(text, ref):
    assert find_reference(text) == ref


def test_potential_conversions():
    assert convert_potential(-0.4, "V vs. Ag/AgCl", "V vs. NHE") == pytest.approx(-0.203)
    assert convert_potential(-0.5, "V vs. NHE", "V vs. RHE", pH=7) == pytest.approx(-0.0863)
    assert convert_potential(1.0, "V vs. RHE", "mV vs. NHE", pH=14) == pytest.approx(172.6)
    assert potential_shift("SCE", "SCE") == 0.0


def test_rhe_needs_ph():
    with pytest.raises(NeedsValue) as e:
        convert_potential(1.0, "V vs. RHE", "V vs. NHE")
    assert e.value.name == "pH"


# ------------------------------------------------------------ step 2b/c: rates, catalyst mass, molar mass
def test_molar_mass():
    assert molar_mass("H2") == pytest.approx(2.016)
    assert molar_mass("g-C3N4") == pytest.approx(92.06, rel=1e-3)
    with pytest.raises(UnitError):
        molar_mass("Xx2")


def test_split_amount():
    assert split_amount("mg g-1 h-1") == ("mg", "g-1 h-1")
    assert split_amount("µmol/h") == ("µmol", "/h")


@pytest.mark.parametrize("frm, species, mass, value, expected", [
    ("mmol g-1 h-1", None, None, 6.0, 6000.0),                 # plain
    ("mg g-1 h-1", "H2", None, 1.0, 496.03),                   # molar mass of H2
    ("µmol h-1", None, (50, "mg"), 300, 6000.0),               # per gram of catalyst
    ("µmol/h", None, (0.05, "g"), 300, 6000.0),
    ("mg h-1", "H2", (50, "mg"), 1.0, 9920.6),                 # both
])
def test_amount_rate_converter(frm, species, mass, value, expected):
    fn, _ = amount_rate_converter(frm, "µmol g-1 h-1", species, mass)
    assert fn(value) == pytest.approx(expected, rel=1e-4)


def test_amount_rate_converter_reports_what_is_missing():
    with pytest.raises(NeedsValue) as e:
        amount_rate_converter("µmol h-1", "µmol g-1 h-1")
    assert e.value.name == "catalyst mass"
    with pytest.raises(UnitError, match="species"):
        amount_rate_converter("mg g-1 h-1", "µmol g-1 h-1")


# ------------------------------------------------------------ standardise with dictionary entries
BAND_GAP = {"unit": "eV", "contexts": ["spectroscopy"]}
CB = {"unit": "V", "reference": "NHE"}
H2 = {"unit": "µmol g-1 h-1", "species": "H2"}


def _res(**kw):
    return {"status": "verified", "value": None, "range": "", "error": "", "unit": "", "evidence": "", **kw}


def test_standardise_band_gap_from_wavelength():
    s = standardise(_res(value=455.8, unit="nm", error="± 2"), BAND_GAP)
    assert s["standard_value"] == pytest.approx(2.7201, rel=1e-4) and s["standard_unit"] == "eV"
    assert "E = hc/λ" in s["conversion_note"] and s["standard_error"].startswith("± 0.0119")


def test_standardise_potential_to_nhe():
    s = standardise(_res(value=-0.4, unit="V vs. Ag/AgCl"), CB)
    assert (s["standard_value"], s["standard_unit"]) == (pytest.approx(-0.203), "V vs. NHE")
    assert standardise(_res(value=-1.1, unit="V vs. NHE"), CB)["conversion_note"] == "Already in V vs. NHE."


def test_standardise_reference_found_in_evidence():
    r = _res(value=-1.1, unit="V", evidence="the flat band potential is -1.1 V vs. SCE")
    assert standardise(r, CB)["standard_value"] == pytest.approx(-0.859)


def test_standardise_rhe_asks_for_ph_then_uses_it():
    r = _res(value=-0.5, unit="V vs. RHE", range="-0.6 to -0.4")
    first = standardise(r, CB)
    assert first["needs"] == ["pH"] and first["standard_value"] is None
    second = standardise(r, CB, {"pH": 7.0})
    assert second["standard_value"] == pytest.approx(-0.9137)
    assert second["standard_range"] == "-1.0137 to -0.8137" and "pH 7" in second["conversion_note"]


def test_standardise_no_reference_stated():
    assert "not stated" in standardise(_res(value=-1.1, unit="V"), CB)["conversion_note"]


def test_standardise_rate_asks_for_catalyst_mass_then_uses_it():
    r = _res(value=300, unit="µmol h-1", error="± 15")
    assert standardise(r, H2)["needs"] == ["catalyst mass"]
    s = standardise(r, H2, {"catalyst mass": (50.0, "mg")})
    assert (s["standard_value"], s["standard_error"]) == (6000.0, "± 300")


def test_standardise_rate_in_mg_uses_molar_mass():
    s = standardise(_res(value=1.0, unit="mg g-1 h-1"), H2)
    assert s["standard_value"] == pytest.approx(496.03, rel=1e-4) and "molar mass of H2" in s["conversion_note"]


# ------------------------------------------------------------ dictionary validation
@pytest.mark.parametrize("entry, msg", [
    ({"contexts": ["magic"]}, "Unknown context"),
    ({"reference": "XYZ"}, "Unknown reference"),
    ({"species": "Xx2"}, "species"),
])
def test_add_parameter_validates_new_fields(tmp_path, entry, msg):
    with pytest.raises(ValueError, match=msg):
        add_parameter("thing", entry, str(tmp_path / "p.json"))


# ------------------------------------------------------------ tools
def test_check_plausible_uses_entry_contexts():
    t = PaperTools("x", {"name": "band gap", "parameter": "band gap"}, {"band gap": {**BAND_GAP,
                                                                                    "plausible_range": [1, 4.5]}})
    assert t.check_plausible(455.8, "nm")["plausible"] is True
    assert t.check_plausible(100, "nm")["plausible"] is False          # = 12.4 eV


def test_convert_unit_tool_potentials_and_molar_mass():
    t = PaperTools("x", {"name": "x"}, {})
    assert t.run("convert_unit", {"value": 1.0, "from_unit": "V vs. RHE", "to_unit": "V vs. NHE",
                                  "pH": 7})["value"] == pytest.approx(0.5863)
    assert "pH" in t.run("convert_unit", {"value": 1.0, "from_unit": "V vs. RHE", "to_unit": "V vs. NHE"})["error"]
    assert t.run("molar_mass", {"formula": "H2"})["molar_mass"] == pytest.approx(2.016)


# ------------------------------------------------------------ agent: extracts the values it needs
PARAMS = {
    "H2 evolution rate": H2,
    "conduction band potential": CB,
    "catalyst mass": {"unit": "mg", "plausible_range": [0.01, 10000]},
    "electrolyte pH": {"unit": "", "plausible_range": [0, 14]},
}
PAPER = ("DCN produced H2 at 300 µmol h-1. In each test, 50 mg of photocatalyst was dispersed in 100 mL water. "
         "Mott-Schottky plots in 0.5 M Na2SO4 (pH = 6.8) give a flat band potential of -0.95 V vs. RHE for DCN.")


def test_agent_extracts_supporting_values_then_converts(monkeypatch, tmp_path):
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "H2 evolution rate"},
                                               {"name": "conduction band potential"}]})),
        _msg(_Call("save_result", {"sample_name": "DCN", "value": 300, "unit": "µmol h-1", "evidence": "DCN produced H2 at 300 µmol h-1"})),
        _msg(_Call("save_result", {"sample_name": "DCN", "value": -0.95, "unit": "V vs. RHE",
                                   "evidence": "flat band potential of -0.95 V vs. RHE for DCN"})),
        # supporting runs, in the order the needs were found
        _msg(_Call("save_result", {"sample_name": "DCN", "value": 50, "unit": "mg", "evidence": "50 mg of photocatalyst was dispersed"})),
        _msg(_Call("save_result", {"sample_name": "DCN", "value": 6.8, "unit": "", "evidence": "0.5 M Na2SO4 (pH = 6.8)"})),
    ]
    chat = _fake_chat(replies)
    monkeypatch.setattr(agent, "chat", chat)

    final = agent.run_agent(PAPER, "p", "H2 evolution rate and conduction band potential", params=PARAMS)

    assert replies == []
    assert "Extract this data item now: catalyst mass" in chat.sent[3][-1]["content"]
    assert "Extract this data item now: electrolyte pH" in chat.sent[4][-1]["content"]

    by_item = {r["item"]: r for r in final["results"]}
    assert by_item["H2 evolution rate"]["standard_value"] == 6000.0
    assert by_item["conduction band potential"]["standard_value"] == pytest.approx(-1.35188)
    assert by_item["conduction band potential"]["standard_unit"] == "V vs. NHE"
    assert by_item["catalyst mass"]["supporting_for"] == "H2 evolution rate"
    assert by_item["electrolyte pH"]["supporting_for"] == "conduction band potential"

    with open(tmp_path / "p_extraction.csv", encoding="utf-8-sig") as f:
        rows = {r["item"]: r for r in csv.DictReader(f)}
    assert rows["H2 evolution rate"]["standard_value"] == "6000.0"
    assert rows["catalyst mass"]["supporting_for"] == "H2 evolution rate"


def test_agent_reuses_value_the_user_already_asked_for(monkeypatch):
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "H2 evolution rate"}, {"name": "catalyst mass"}]})),
        _msg(_Call("save_result", {"sample_name": "DCN", "value": 300, "unit": "µmol h-1", "evidence": "DCN produced H2 at 300 µmol h-1"})),
        _msg(_Call("save_result", {"sample_name": "DCN", "value": 50, "unit": "mg", "evidence": "50 mg of photocatalyst was dispersed"})),
    ]
    monkeypatch.setattr(agent, "chat", _fake_chat(replies))

    final = agent.run_agent(PAPER, "p", "H2 rate and catalyst mass", params=PARAMS)

    assert replies == []                                    # no extra run
    assert final["results"][0]["standard_value"] == 6000.0
    assert len(final["results"]) == 2


def test_supporting_value_not_found_leaves_note(monkeypatch):
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "H2 evolution rate"}]})),
        _msg(_Call("save_result", {"sample_name": "DCN", "value": 300, "unit": "µmol h-1", "evidence": "DCN produced H2 at 300 µmol h-1"})),
        _msg(_Call("save_result", {"value": None, "evidence": ""})),         # catalyst mass not reported
    ]
    monkeypatch.setattr(agent, "chat", _fake_chat(replies))

    final = agent.run_agent(PAPER, "p", "H2 rate", params=PARAMS)

    r = final["results"][0]
    assert r["standard_value"] is None and "needs the catalyst mass" in r["conversion_note"]


# ------------------------------------------------------------ custom "electron" context (eV vs. NHE)
def test_electron_context_reads_ev_as_volts():
    assert convert(1.94, "eV", "V", ["electron"]) == pytest.approx(1.94)
    s = standardise(_res(value=1.94, unit="eV vs NHE"), {"unit": "V", "reference": "NHE", "contexts": ["electron"]})
    assert (s["standard_value"], s["standard_unit"]) == (pytest.approx(1.94), "V vs. NHE")
    assert "energy per electron" in s["conversion_note"]
    s = standardise(_res(value=1.94, unit="eV vs. RHE"), {"unit": "V", "reference": "NHE", "contexts": ["electron"]},
                    {"pH": 7.0})
    assert s["standard_value"] == pytest.approx(1.5263)
