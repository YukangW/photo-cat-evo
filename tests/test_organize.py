"""Offline tests for organize.py (one row per sample, one column per parameter)."""
import csv

import pytest

from PhotoCatEvo.organize import cell_value, organize, range_to_number

COLUMNS = ["paper", "doi", "item", "parameter", "sample_name", "is_main", "value", "range", "unit",
           "standard_value", "standard_range", "standard_unit", "status"]


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def _read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("text, expected", [
    ("2.6-2.8", (2.7, "range average")),
    ("2.6–2.8", (2.7, "range average")),          # en dash
    ("2.6 to 2.8", (2.7, "range average")),
    ("-0.5 to -0.3", (-0.4, "range average")),          # negative potentials
    (">160", (160.0, "bound")),
    ("<5", (5.0, "bound")),
    ("≥ 3.1", (3.1, "bound")),
    ("1,205-1,300", (1252.5, "range average")),         # thousands separators
    ("", (None, "")),
    ("not given", (None, "")),
])
def test_range_to_number(text, expected):
    assert range_to_number(text) == expected


def test_cell_value_prefers_standard_units():
    assert cell_value({"value": "6.0", "unit": "mmol g-1 h-1", "standard_value": "6000",
                       "standard_unit": "µmol g-1 h-1"}) == (6000.0, "µmol g-1 h-1", "value")
    assert cell_value({"value": "0", "standard_value": "0", "standard_unit": "eV"}) == (0.0, "eV", "value")
    assert cell_value({"value": "air"}) == ("air", "", "text")
    assert cell_value({"range": ">160", "unit": "m2 g-1", "standard_range": ">160",
                       "standard_unit": "m2 g-1"}) == (160.0, "m2 g-1", "bound")
    assert cell_value({"range": "2.6-2.8", "unit": "eV"}) == (2.7, "eV", "range average (paper unit)")


def test_organize_one_row_per_sample(tmp_path):
    _write(tmp_path / "paper_021_extraction.csv", [
        {"paper": "paper_021", "doi": "10.1016/j.carbon.2024.118841", "item": "band gap", "sample_name": "CNU20",
         "is_main": "yes", "value": "2.33", "unit": "eV", "standard_value": "2.33", "standard_unit": "eV",
         "status": "verified"},
        {"paper": "paper_021", "doi": "10.1016/j.carbon.2024.118841", "item": "band gap", "sample_name": "g-C3N4",
         "is_main": "no", "value": "2.41", "unit": "eV", "standard_value": "2.41", "standard_unit": "eV",
         "status": "verified"},
        {"paper": "paper_021", "doi": "10.1016/j.carbon.2024.118841", "item": "BET surface area",
         "sample_name": "CNU 20", "is_main": "yes", "range": ">160", "unit": "m2 g-1", "standard_range": ">160",
         "standard_unit": "m2 g-1", "status": "range_only"},                         # same sample, other spelling
        {"paper": "paper_021", "item": "C:N ratio", "sample_name": "CNU20", "is_main": "yes", "value": "1.77",
         "standard_value": "1.77", "status": "verified"},                             # dimensionless
        {"paper": "paper_021", "item": "heating time", "status": "not_reported"},     # skipped
    ])
    _write(tmp_path / "paper_022_extraction.csv", [
        {"paper": "paper_022", "doi": "10.1/x", "item": "band gap", "sample_name": "TiO2-500", "is_main": "yes",
         "range": "3.0-3.2", "unit": "eV", "standard_range": "3.0-3.2", "standard_unit": "eV",
         "status": "range_only"},
        {"paper": "paper_022", "doi": "10.1/x", "item": "atmosphere", "sample_name": "TiO2-500", "is_main": "yes",
         "value": "air", "status": "verified"},
    ])
    out = organize(str(tmp_path), str(tmp_path / "all.csv"), with_source=True)
    rows = _read(out)

    assert list(rows[0].keys())[:7] == ["paper", "doi", "sample_name",
                                        "band gap", "band gap_unit", "band gap_source", "BET surface area"]
    assert "is_main" not in rows[0]
    assert "atmosphere" in rows[0] and "atmosphere_unit" not in rows[0]      # text: no unit column
    assert "C:N ratio" in rows[0] and "C:N ratio_unit" not in rows[0]        # dimensionless: no unit column
    table = {(r["paper"], r["sample_name"]): r for r in rows}
    assert len(rows) == 3
    cnu = table[("paper_021", "CNU20")]
    assert (cnu["band gap"], cnu["band gap_unit"], cnu["C:N ratio"]) == ("2.33", "eV", "1.77")
    assert (cnu["BET surface area"], cnu["BET surface area_source"]) == ("160.0", "bound")
    assert table[("paper_021", "g-C3N4")]["BET surface area"] == ""       # not reported for this sample
    tio2 = table[("paper_022", "TiO2-500")]
    assert (tio2["band gap"], tio2["band gap_source"], tio2["atmosphere"]) == ("3.1", "range average", "air")
    assert tio2["doi"] == "10.1/x"


def test_organize_needs_files(tmp_path):
    with pytest.raises(SystemExit):
        organize(str(tmp_path))


def test_organize_finds_csvs_in_subfolders(tmp_path):
    a = tmp_path / "batch1" / "paper_021.pdf"
    b = tmp_path / "batch2" / "sub" / "paper_022.pdf"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    _write(a / "paper_021_extraction.csv", [
        {"paper": "paper_021.pdf", "doi": "10.1/a", "item": "band gap", "sample_name": "CNU20", "is_main": "yes",
         "value": "2.33", "unit": "eV", "standard_value": "2.33", "standard_unit": "eV", "status": "verified"}])
    _write(b / "paper_022_extraction.csv", [
        {"paper": "sub/paper_022.pdf", "doi": "10.1/b", "item": "band gap", "sample_name": "TiO2", "is_main": "yes",
         "value": "3.1", "unit": "eV", "standard_value": "3.1", "standard_unit": "eV", "status": "verified"}])
    rows = _read(organize(str(tmp_path), str(tmp_path / "all.csv")))
    assert sorted(r["sample_name"] for r in rows) == ["CNU20", "TiO2"]
    with pytest.raises(SystemExit):                                  # top folder only: nothing there
        organize(str(tmp_path), str(tmp_path / "x.csv"), recursive=False)


def test_organize_warns_when_a_paper_is_in_two_files(tmp_path, capsys):
    for run in ("run1", "run2"):
        d = tmp_path / run
        d.mkdir()
        _write(d / "p_extraction.csv", [
            {"paper": "p", "doi": "10.1/a", "item": "band gap", "sample_name": "CN", "is_main": "yes",
             "value": "2.7" if run == "run1" else "2.8", "unit": "eV", "status": "verified"}])
    rows = _read(organize(str(tmp_path), str(tmp_path / "all.csv")))
    assert len(rows) == 1 and rows[0]["band gap"] == "2.7"            # first file wins
    out = capsys.readouterr().out
    assert "is in 2 files" in out and "kept 2.7, ignored 2.8" in out
