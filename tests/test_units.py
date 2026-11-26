"""Offline tests for Pint unit conversion, the convert_unit tool and the standardisation step.

Run:  python -m pytest -q
"""
import csv

import pytest

from PhotoCatEvo import agent
from PhotoCatEvo.parameters import add_parameter, load_parameters, standard_unit_for
from PhotoCatEvo.standardise import standardise
from PhotoCatEvo.tools import PaperTools
from PhotoCatEvo.units import UnitError, convert, convert_error, convert_text, parse_unit, si_unit
from test_agent import _Call, _fake_chat, _msg, _tmp_results  # noqa: F401  (fixture: results go to tmp)


# ------------------------------------------------------------ reading units as papers write them
@pytest.mark.parametrize("written, pint_form", [
    ("m2 g-1", "m²/g"), ("m2 g−1", "m²/g"), ("m²/g", "m²/g"), ("m² g⁻¹", "m²/g"),
    ("µmol g-1 h-1", "µmol/g/h"), ("μmol h−1 g−1", "µmol/g/h"), ("umol/g/h", "µmol/g/h"),
    ("mmol·g⁻¹·h⁻¹", "mmol/g/h"), ("µA cm-2", "µA/cm²"), ("cm3 g-1", "cm³/g"),
    ("eV", "eV"), ("nm", "nm"), ("Å", "Å"), ("%", "%"), ("V vs. NHE", "V"), ("°C", "°C"),
])
def test_parse_unit(written, pint_form):
    assert parse_unit(written) == parse_unit(pint_form)


@pytest.mark.parametrize("bad", ["a.u.", "arb. units", "furlongs per fortnight"])
def test_parse_unit_rejects_unreadable(bad):
    with pytest.raises(UnitError):
        parse_unit(bad)


# ------------------------------------------------------------ converting
@pytest.mark.parametrize("value, frm, to, expected", [
    (5e-9, "m", "nm", 5.0),
    (12, "Å", "nm", 1.2),
    (6.0, "mmol g−1 h−1", "µmol g-1 h-1", 6000.0),
    (1.5, "mA cm-2", "µA cm-2", 1500.0),
    (0.45, "m2/kg", "m2 g-1", 0.00045),
    (25, "°C", "K", 298.15),
])
def test_convert(value, frm, to, expected):
    assert convert(value, frm, to) == pytest.approx(expected)


def test_convert_refuses_different_kinds_of_quantity():
    with pytest.raises(UnitError, match="measure different things"):
        convert(2.7, "eV", "nm")


def test_errors_are_converted_as_differences():
    assert convert_error(0.3, "mmol g-1 h-1", "µmol g-1 h-1") == 300.0
    assert convert_error(2, "°C", "K") == 2.0                     # no +273.15 for an uncertainty


def test_convert_text_keeps_range_symbols_and_signs():
    assert convert_text(">160", "m2 g-1", "m2 kg-1") == ">160000"
    assert convert_text("2.6-2.8", "nm", "Å") == "26-28"          # dash between numbers is not a minus
    assert convert_text("-0.5 to -0.3", "V", "mV") == "-500 to -300"
    assert convert_text("± 0.3", "mmol g-1 h-1", "µmol g-1 h-1", as_error=True) == "± 300"


def test_si_unit():
    assert si_unit("ns") == "s"
    assert parse_unit(si_unit("m2 g-1")) == parse_unit("m2 kg-1")


# ------------------------------------------------------------ the convert_unit tool
def test_convert_unit_tool():
    t = PaperTools("text", {"name": "x"}, {})
    assert t.run("convert_unit", {"value": 5e-9, "from_unit": "m", "to_unit": "nm"})["value"] == 5.0
    assert t.run("convert_unit", {"value": 1, "from_unit": "eV", "to_unit": "nm"})["value"] == pytest.approx(1239.84, rel=1e-5)
    assert "error" in t.run("convert_unit", {"value": 1, "from_unit": "eV", "to_unit": "kg"})
    assert "error" in t.run("convert_unit", {"value": 1, "from_unit": "a.u.", "to_unit": "nm"})


# ------------------------------------------------------------ standardise one result
def _res(**kw):
    return {"status": "verified", "value": None, "range": "", "error": "", "unit": "", **kw}


def test_standardise_to_dictionary_unit():
    s = standardise(_res(value=6.0, unit="mmol g−1 h−1", error="± 0.3"), {"unit": "µmol g-1 h-1"})
    assert (s["standard_value"], s["standard_error"], s["standard_unit"]) == (6000.0, "± 300", "µmol g-1 h-1")
    assert "Converted from mmol g−1 h−1" in s["conversion_note"]


def test_standardise_range_only():
    s = standardise(_res(status="range_only", range=">160", unit="m2 g−1"), {"unit": "m2 g-1"})
    assert (s["standard_value"], s["standard_range"]) == (None, ">160")
    assert s["conversion_note"] == "Already in m2 g-1."


def test_standardise_not_in_dictionary_uses_si():
    s = standardise(_res(value=8.47, unit="ns"), None)
    assert (s["standard_value"], s["standard_unit"]) == (pytest.approx(8.47e-9), "s")


def test_standardise_dimensionless_not_in_dictionary():
    assert standardise(_res(value=0.86), None)["standard_value"] == 0.86            # C/N ratio, no unit
    assert standardise(_res(value=4.2, unit="%"), None)["standard_unit"] == "%"   # % stays %


def test_standardise_keeps_reference_electrode_in_note():
    s = standardise(_res(value=-1.12, unit="V vs. NHE"), {"unit": "V"})
    assert s["standard_value"] == -1.12 and "vs. NHE" in s["conversion_note"]


@pytest.mark.parametrize("result, std_unit, note", [
    (_res(value=2.7, unit=""), "eV", "No unit reported"),            # dictionary expects a unit
    (_res(value="anatase"), None, "Text value"),
    (_res(value=2.7, unit="eV"), "nm", "measure different things"),  # no spectroscopy context allowed
    (_res(value=3.0, unit="a.u."), None, "arbitrary"),
])
def test_standardise_cannot_convert(result, std_unit, note):
    s = standardise(result, {"unit": std_unit} if std_unit else None)
    assert s["standard_value"] is None and note in s["conversion_note"]


def test_standardise_skips_failed_and_not_reported():
    assert standardise({"status": "not_reported", "value": None}, {"unit": "eV"})["standard_unit"] == ""


# ------------------------------------------------------------ dictionary
def test_shipped_dictionary_units_all_readable():
    for name, entry in load_parameters().items():
        parse_unit(standard_unit_for(entry))      # raises if Pint cannot read it


def test_add_parameter_rejects_unreadable_unit(tmp_path):
    with pytest.raises(ValueError, match="unit"):
        add_parameter("intensity", {"unit": "a.u."}, str(tmp_path / "p.json"))


def test_standard_unit_overrides_unit():
    assert standard_unit_for({"unit": "nm", "standard_unit": "m"}) == "m"
    assert standard_unit_for({"unit": "nm"}) == "nm"
    assert standard_unit_for(None) == ""


# ------------------------------------------------------------ end to end: runs AFTER extraction
def test_agent_adds_standard_columns_after_extraction(monkeypatch, tmp_path):
    params = {"H2 evolution rate": {"unit": "µmol g-1 h-1", "plausible_range": [0.01, 200000]},
              "pore size": {"unit": "nm"}}
    paper = ("DCN gave an H2 evolution rate of 6.0 mmol g−1 h−1 (± 0.3). "
             "The average pore size of DCN is 38 Å. The fluorescence lifetime of DCN is 8.47 ns.")
    replies = [
        _msg(_Call("set_targets", {"targets": [{"name": "H2 evolution rate"}, {"name": "pore size"},
                                               {"name": "fluorescence lifetime"}]})),
        _msg(_Call("save_result", {"value": 6.0, "unit": "mmol g−1 h−1", "error": "± 0.3", "sample_name": "DCN",
                                   "evidence": "H2 evolution rate of 6.0 mmol g−1 h−1 (± 0.3)"})),
        _msg(_Call("save_result", {"value": 38, "unit": "Å", "sample_name": "DCN", "evidence": "average pore size of DCN is 38 Å"})),
        _msg(_Call("save_result", {"value": 8.47, "unit": "ns", "sample_name": "DCN",
                                   "evidence": "fluorescence lifetime of DCN is 8.47 ns"})),
    ]
    chat = _fake_chat(replies)
    monkeypatch.setattr(agent, "chat", chat)

    final = agent.run_agent(paper, "p", "H2 rate, pore size and lifetime", params=params)

    # the extraction saw the paper's own units: nothing was converted before save_result
    assert [r["value"] for r in final["results"]] == [6.0, 38, 8.47]
    assert [r["unit"] for r in final["results"]] == ["mmol g−1 h−1", "Å", "ns"]

    with open(tmp_path / "p_extraction.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [(r["standard_value"], r["standard_error"], r["standard_unit"]) for r in rows] == [
        ("6000.0", "± 300", "µmol g-1 h-1"),
        ("3.8", "", "nm"),
        ("8.47e-09", "", "s"),
    ]
