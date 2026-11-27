"""Offline tests for evaluate.py (ground truth in the team's sheet format vs PhotoCatEvo-table output)."""
import csv

import pytest

from PhotoCatEvo import evaluate as E
from PhotoCatEvo.organize import organize
from PhotoCatEvo.parameters import load_parameters

GT_HEADER = ["Paper", "DOI", "Number of C3N4 photocatalysts discussed", "Sample Name",
             "Specific Surface Area", "Specific Surface Area Units", "Band Gap", "Band Gap Units", "C:N ratio",
             "Hydrogen Evolution Rate", "Hydrogen Evolution Rate Units", "Temperature", "Temperature Units",
             "Atmosphere used", "Notes"]
GT_ROWS = [
    ["p021", "https://doi.org/10.1016/j.carbon.2024.118841", 2, "g-C3N4", 18.06, "m2/g", 2.41, "eV", 1.57,
     "", "", 550, "°C", "N2", ""],
    ["p021", "10.1016/j.carbon.2024.118841", 2, "CNU20", 20, "m2/g", 2.33, "eV", 1.77,
     410.51, "µmol/g/h", 160, "°C", "", "H2 on Pt-CNU20"],
    ["p017", "10.1016/j.mssp.2023.107459", 2, "3NBCN", 22.8, "m2/g", 2.7, "eV", 0.76, 1764.9, "µmol/g/h",
     "", "", "", ""],
    ["p017", "10.1016/j.mssp.2023.107459", 2, "GCN", 22.5, "m2/g", 2.75, "eV", 0.88, 206.3, "µmol/g/h",
     "", "", "", ""],
    ["p099", "10.9999/not.extracted", 1, "X", 1, "m2/g", 3.0, "eV", "", "", "", "", "", "", ""],
]
PRED_HEADER = ["paper", "doi", "sample_name", "BET surface area", "BET surface area_unit", "band gap",
               "band gap_unit", "C:N ratio", "H2 evolution rate", "H2 evolution rate_unit",
               "heating temperature", "heating temperature_unit", "atmosphere"]
PRED_ROWS = [
    ["paper_021", "10.1016/j.carbon.2024.118841", "CNU20", 20, "m2 g-1", 2.33, "eV", 1.77, "", "",
     433.15, "K", ""],
    ["paper_021", "10.1016/j.carbon.2024.118841", "Pt–CNU20", "", "", "", "", "", 0.41051, "mmol g-1 h-1",
     "", "", ""],                                                       # activity on the loaded sample
    ["paper_021", "10.1016/j.carbon.2024.118841", "g-C3N4", 18.1, "m2 g-1", 2.45, "eV", "", "", "",
     823.15, "K", "nitrogen"],                                          # 2.45 wrong; C:N missed
    ["paper_017", "10.1016/j.mssp.2023.107459", "3NBCN", 22.8, "m2 g-1", 2.70, "eV", 0.76, 1764.9,
     "µmol g-1 h-1", 500, "°C", ""],                                    # temperature not in GT -> FP
    ["paper_017", "10.1016/j.mssp.2023.107459", "NBCN-5", 30, "m2 g-1", "", "", "", "", "", "", "", ""],  # extra
    ["paper_555", "10.5555/only.predicted", "Y", 1, "m2 g-1", "", "", "", "", "", "", "", ""],
]


def _csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


@pytest.fixture
def files(tmp_path):
    return (_csv(tmp_path / "gt.csv", GT_HEADER, GT_ROWS), _csv(tmp_path / "pred.csv", PRED_HEADER, PRED_ROWS))


def _run(files, **kw):
    params = load_parameters()
    truth, cols = E.load_truth(files[0], params)
    pred = E.load_predictions(files[1], params)
    return E.evaluate(truth, list(cols), pred, **kw), cols


def _labels(res, sample, param):
    return [d["label"] for d in res["details"] if d["true_sample"] == sample and d["parameter"] == param]


def test_columns_are_mapped_through_the_dictionary(files):
    _, cols = _run(files)
    assert list(cols) == ["BET surface area", "band gap", "C:N ratio", "H2 evolution rate",
                          "heating temperature", "atmosphere"]


def test_papers_matched_by_doi(files):
    res, _ = _run(files)
    assert res["papers"] == ["10.1016/j.carbon.2024.118841", "10.1016/j.mssp.2023.107459"]
    assert res["only_truth"] == ["10.9999/not.extracted"] and res["only_pred"] == ["10.5555/only.predicted"]


def test_filter_keeps_only_ground_truth_papers(tmp_path, files):
    pred = _csv(tmp_path / "all.csv", PRED_HEADER, PRED_ROWS + [
        ["paper_777", "", "Z", 5, "m2 g-1", "", "", "", "", "", "", "", ""]])          # no DOI
    s = E.filter_to_truth(files[0], pred, str(tmp_path / "subset.csv"))
    with open(tmp_path / "subset.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert {r["paper"] for r in rows} == {"paper_021", "paper_017"}                  # only GT papers
    assert len(rows) == 5 and list(rows[0].keys()) == PRED_HEADER                    # all columns kept
    assert s["dropped_papers"] == ["paper_555"] and s["papers_without_doi"] == ["paper_777"]
    assert s["truth_papers_not_extracted"] == ["p099"]


def test_filter_default_output_name(tmp_path, files):
    s = E.filter_to_truth(files[0], files[1])
    assert s["output"].endswith("pred_gt_papers.csv")


def test_units_are_converted_before_comparing(files):
    res, _ = _run(files)
    assert _labels(res, "CNU20", "H2 evolution rate") == ["TP"]       # 0.41051 mmol -> 410.51 µmol, from Pt–CNU20
    assert _labels(res, "CNU20", "heating temperature") == ["TP"]     # 433.15 K -> 160 °C
    assert _labels(res, "g-C3N4", "heating temperature") == ["TP"]    # 823.15 K -> 550 °C


def test_labels_like_li_et_al(files):
    res, _ = _run(files)
    assert _labels(res, "g-C3N4", "BET surface area") == ["TP"]       # 18.1 vs 18.06: within 2 %
    assert _labels(res, "g-C3N4", "band gap") == ["FP"]               # 2.45 vs 2.41: more than 0.02 eV
    assert _labels(res, "g-C3N4", "C:N ratio") == ["FN"]              # not extracted
    assert _labels(res, "g-C3N4", "atmosphere") == ["TP"]             # nitrogen == N2
    assert _labels(res, "3NBCN", "heating temperature") == ["FP"]     # extracted, not in the ground truth
    assert _labels(res, "GCN", "band gap") == ["FN"]                  # sample missed completely
    extra = [d["label"] for d in res["details"] if d["predicted_sample"] == "NBCN-5" and d["label"] != "TN"]
    assert extra == ["FP"]                                            # invented sample


def test_matches_record_how(files):
    res, _ = _run(files)
    how = {(m["true_sample"], m["predicted_sample"]): m["match"] for m in res["matches"]}
    assert how[("CNU20", "CNU20")] == "exact" and how[("GCN", "")] == "missed" and how[("", "NBCN-5")] == "extra"
    merged = [m for m in res["matches"] if m["true_sample"] == "CNU20"][0]
    assert merged["merged_from"] == "Pt–CNU20"


def test_metrics_and_strict_mode(files):
    res, _ = _run(files)
    bg = next(m for m in res["metrics"] if m["parameter"] == "band gap")
    assert (bg["TP"], bg["FP"], bg["FN"]) == (2, 1, 1)                # CNU20, 3NBCN | g-C3N4 | GCN
    assert bg["precision"] == pytest.approx(2 / 3) and bg["recall"] == pytest.approx(2 / 3)
    strict, _ = _run(files, strict=True)
    bg = next(m for m in strict["metrics"] if m["parameter"] == "band gap")
    assert (bg["FP"], bg["FN"]) == (1, 2)


def test_without_merging_the_loaded_sample_counts_as_extra(files):
    res, _ = _run(files, merge=False)
    assert _labels(res, "CNU20", "H2 evolution rate") == ["FN"]


def test_whole_chain_from_extraction_csvs(tmp_path, files):
    """extraction CSV -> PhotoCatEvo-table -> PhotoCatEvo-evaluate (main)."""
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    cols = ["paper", "doi", "item", "parameter", "sample_name", "is_main", "value", "range", "unit",
            "standard_value", "standard_range", "standard_unit", "status"]
    _csv(res_dir / "paper_017_extraction.csv", cols, [
        ["paper_017", "10.1016/j.mssp.2023.107459", "band gap", "band gap", "3NBCN", "yes", 2.7, "", "eV",
         2.7, "", "eV", "verified"],
        ["paper_017", "10.1016/j.mssp.2023.107459", "band gap", "band gap", "GCN", "no", 2.75, "", "eV",
         2.75, "", "eV", "verified"]])
    merged = organize(str(res_dir), str(tmp_path / "all.csv"))
    out = tmp_path / "eval"
    E.main([files[0], merged, "--out", str(out)])
    with open(out / "metrics.csv", encoding="utf-8-sig") as f:
        bg = next(r for r in csv.DictReader(f) if r["parameter"] == "band gap")
    assert (bg["TP"], bg["FP"], bg["FN"]) == ("2", "0", "0")
