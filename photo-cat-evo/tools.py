"""
Tools the agent can call. Plain Python, no LLM.

Planning (once per prompt):
  set_targets(targets)            -> the list of separate data items the prompt asks for

Extraction (once per data item):
  search_text(keyword)            -> sentences of the paper containing the keyword
  check_evidence(quote)           -> is this quote really in the paper?
  check_plausible(value, unit)    -> is this value in the expected range (from the parameter dictionary)?
  convert_unit(value, from, to)   -> convert a value between units with Pint (e.g. m -> nm)
  save_result(...)                -> final answer; runs the checks and REJECTS it if they fail
"""
import re
import unicodedata

from parameters import find_parameter, load_parameters
from units import UnitError, convert


def _normalize(s: str) -> str:
    """Make text comparable: ligatures, invisible characters, dashes, quotes, case and spaces."""
    s = unicodedata.normalize("NFKC", s)                       # ligatures (ﬁ -> fi), subscripts, full-width chars
    s = re.sub(r"[\u00ad\u200b-\u200f\u2060\ufeff]", "", s)    # soft hyphens, zero-width characters
    s = s.lower()
    for dash in ("\u2212", "\u2013", "\u2014", "\u2010", "\u2011"):
        s = s.replace(dash, "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    # remove ALL whitespace: PDFs often split formulas ("C3 N4 -VI") or lines unpredictably,
    # while the model usually writes them without spaces ("C3N4-VI")
    return re.sub(r"\s+", "", s)


def _search_normalize(s: str) -> str:
    """Like _normalize, but keeps single spaces (used to search for keywords)."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u00ad\u200b-\u200f\u2060\ufeff]", "", s)
    for dash in ("\u2212", "\u2013", "\u2014", "\u2010", "\u2011"):
        s = s.replace(dash, "-")
    return re.sub(r"\s+", " ", s.lower()).strip()


# Signs that a quote reports a value from ANOTHER paper (introduction, comparison tables, cited work)
_CITATION_PATTERNS = [
    (r"\[\s*\d+(?:\s*[,\u2013\u2212-]\s*\d+)*\s*\]", "a reference number like [12]"),
    (r"\bet\s+al\b", "'et al.'"),
    (r"\bref(?:erence)?s?\.?\s*\[?\d", "a reference ('Ref. 12')"),
    (r"\bpreviously\s+reported\b|\breported\s+(?:previously|earlier|in\s+the\s+literature)\b",
     "'previously reported'"),
    (r"\bin\s+the\s+literature\b|\bliterature\s+values?\b", "'literature'"),
]


def _literature_reason(quote: str) -> str:
    """'' if the quote looks like this paper's own result, else why it looks like a value from other work."""
    for pattern, what in _CITATION_PATTERNS:
        if re.search(pattern, quote or "", re.IGNORECASE):
            return what
    return ""


def _numbers_in(text: str) -> list[float]:
    """All numbers in a piece of text. Handles thousands separators ("1,250") and minus signs."""
    text = text.replace("−", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _number_in(value: float, text: str) -> bool:
    """Is this number written in the text? Sign is ignored, because in ranges like "2.6-2.8"
    the dash would otherwise be read as a minus sign."""
    return any(abs(abs(n) - abs(value)) < 1e-6 for n in _numbers_in(text))


def plausible_range(target_name: str, params: dict | None = None):
    """(min, max, unit) for this kind of data from the parameter dictionary, or None if it has no range."""
    match = find_parameter(target_name, params)
    if not match or not match[1].get("plausible_range"):
        return None
    lo, hi = match[1]["plausible_range"]
    return float(lo), float(hi), match[1].get("unit", "")


# ============================================================ planning
class PlanTools:
    """Holds the list of data items the planner pulls out of the prompt."""

    def __init__(self, max_targets: int):
        self.max_targets = max_targets
        self.targets = None

    def set_targets(self, targets: list) -> dict:
        clean = []
        for t in targets or []:
            if isinstance(t, str):
                t = {"name": t}
            if not isinstance(t, dict) or not str(t.get("name", "")).strip():
                continue
            clean.append({"name": str(t["name"]).strip(),
                          "unit": str(t.get("unit") or "").strip(),
                          "details": str(t.get("details") or "").strip()})
        if not clean:
            return {"saved": False, "problem": "The list was empty. Give at least one data item."}
        self.targets = clean[: self.max_targets]
        return {"saved": True, "n_targets": len(self.targets)}

    def run(self, name: str, args: dict) -> dict:
        if name != "set_targets":
            return {"error": f"Unknown tool '{name}'. Call set_targets."}
        try:
            return self.set_targets(**args)
        except TypeError as e:
            return {"error": f"Wrong arguments for set_targets: {e}"}


PLAN_TOOLS = [
    {"type": "function", "function": {
        "name": "set_targets",
        "description": "Record the separate pieces of data the user's request asks for, in the order asked.",
        "parameters": {"type": "object",
                       "properties": {"targets": {
                           "type": "array",
                           "items": {"type": "object",
                                     "properties": {
                                         "name": {"type": "string",
                                                  "description": "Short name of the quantity, e.g. 'band gap'."},
                                         "unit": {"type": "string",
                                                  "description": "Expected unit, e.g. 'eV', or '' if none."},
                                         "details": {"type": "string",
                                                     "description": "Anything the user specified about it: "
                                                                    "which sample, conditions, method."}},
                                     "required": ["name"]}}},
                       "required": ["targets"]}}},
]


# ============================================================ extraction
class PaperTools:
    def __init__(self, text: str, target: dict | None = None, params: dict | None = None):
        self.text = text
        self.norm_text = _normalize(text)
        self.target = target or {"name": "band gap", "unit": "eV"}
        params = load_parameters() if params is None else params
        # use the dictionary entry the agent matched, else look the item's name up
        self.range = plausible_range(self.target.get("parameter") or self.target["name"], params)
        self.result = None          # filled by save_result

    # ---------------------------------------------------------------- tools
    def search_text(self, keyword: str, max_hits: int = 8) -> dict:
        """Return sentences that contain the keyword. Ignores case and the way PDFs write dashes,
        ligatures and soft hyphens ("Pt-CNU20" finds "Pt–CNU20"); if nothing is found, spaces are
        ignored too ("C3N4" finds "C3 N4")."""
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", self.text))]
        key = _search_normalize(keyword)
        hits = [x for x in sentences if key in _search_normalize(x)]
        if not hits:
            key_nospace = key.replace(" ", "")
            hits = [x for x in sentences if key_nospace in _search_normalize(x).replace(" ", "")]
        return {"keyword": keyword, "n_hits": len(hits), "sentences": hits[:max_hits]}

    def check_evidence(self, quote: str) -> dict:
        """Check that the quote appears in the paper (ignoring case/spacing/dash differences)."""
        q = _normalize(quote)
        if len(q) < 8:
            return {"found": False, "reason": "Quote too short. Copy a full phrase from the paper."}
        found = q in self.norm_text
        return {"found": found,
                "reason": "Quote found in paper." if found else
                          "Quote NOT found. Copy the text exactly as written in the paper."}

    def check_plausible(self, value: float, unit: str = "") -> dict:
        name = self.target["name"]
        if self.range is None:
            return {"plausible": True,
                    "reason": f"No range is defined for '{name}', so only the evidence is checked."}
        lo, hi, range_unit = self.range
        shown = f"{value} {unit or range_unit}".strip()
        if unit and range_unit:                   # compare in the range's unit, e.g. mmol -> µmol
            try:
                value = convert(value, unit, range_unit)
            except UnitError as e:
                return {"plausible": True,
                        "reason": f"Could not compare with the expected range in {range_unit} ({e}), "
                                  f"so the range was not checked."}
            if shown != f"{value} {range_unit}":
                shown += f" (= {value} {range_unit})"
        ok = lo <= value <= hi
        return {"plausible": ok,
                "reason": f"{shown} is within {lo}-{hi} {range_unit}." if ok else
                          f"{shown} is outside the expected range for {name} ({lo}-{hi} {range_unit}). "
                          f"Check the unit or the sample."}

    def convert_unit(self, value: float, from_unit: str, to_unit: str) -> dict:
        """Convert a value between units with Pint, e.g. 5e-9 m -> 5 nm, 6 mmol g-1 h-1 -> 6000 µmol g-1 h-1."""
        try:
            return {"value": convert(value, from_unit, to_unit), "unit": to_unit,
                    "from": f"{value} {from_unit}"}
        except (UnitError, ValueError) as e:
            return {"error": str(e)}

    def _check_extra(self, label: str, text: str, quote: str) -> list[str]:
        """Check a range or error string: its quote must be in the paper and contain all its numbers."""
        problems = []
        ev = self.check_evidence(quote)
        if not ev["found"]:
            problems.append(f"{label} evidence: {ev['reason']}")
        numbers = _numbers_in(text)
        if not numbers:
            problems.append(f"The {label} '{text}' contains no number.")
        missing = [n for n in numbers if not _number_in(n, quote)]
        if missing:
            problems.append(f"The {label} '{text}' has number(s) {missing} that are not in its evidence quote. "
                            f"Copy the {label} exactly as written in the paper.")
        return problems

    def _check_one(self, label: str, value, unit: str, evidence: str, sample_name: str,
                   range: str, range_evidence: str, error: str, error_evidence: str) -> tuple[list, list]:
        """All checks for ONE sample's value. Returns (problems, warnings)."""
        problems, warnings = [], []
        if not sample_name.strip():
            problems.append(f"{label}: give the sample name exactly as written in the paper "
                            f"(e.g. 'CN-550', 'P25', 'bulk g-C3N4').")
        elif _normalize(sample_name) not in self.norm_text:
            problems.append(f"{label}: sample name '{sample_name}' does not appear in the paper. "
                            f"Copy it exactly as the paper writes it.")
        elif _normalize(sample_name) not in _normalize(evidence + " " + range_evidence):
            warnings.append(f"{label}: sample name '{sample_name}' is not in the quote; make sure the value "
                            f"really belongs to this sample.")
        for quote in {evidence, range_evidence, error_evidence} - {""}:
            if why := _literature_reason(quote):
                problems.append(f"{label}: the quote contains {why}, so it looks like a value from another "
                                f"paper. Only report values measured on samples made in THIS paper.")
                break
        if value is not None:
            ev = self.check_evidence(evidence)
            if not ev["found"]:
                problems.append(f"{label}: {ev['reason']}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not _number_in(value, evidence):
                    problems.append(f"{label}: the value {value} does not appear in the evidence quote. "
                                    f"Report the number exactly as written in the quote.")
                pl = self.check_plausible(value, unit)
                if not pl["plausible"]:
                    problems.append(f"{label}: {pl['reason']}")
            elif _normalize(str(value)) not in _normalize(evidence):   # text answer, e.g. "anatase"
                problems.append(f"{label}: the text value '{value}' does not appear in the evidence quote.")
        if range:
            problems += [f"{label}: {p}" for p in self._check_extra("range", range, range_evidence)]
        if error:
            problems += [f"{label}: {p}" for p in self._check_extra("error", error, error_evidence)]
        return problems, warnings

    @staticmethod
    def _as_value(value):
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, str):     # the model sometimes sends a number as a string ("3.05")
            try:
                return float(value.strip())
            except ValueError:
                return value.strip()
        return value

    def save_result(self, value=None, unit: str = "", evidence: str = "", sample_name: str = "",
                    method: str = "", other_samples: list | None = None, note: str = "",
                    range: str = "", range_evidence: str = "",
                    error: str = "", error_evidence: str = "", all_values: list | None = None) -> dict:
        """Final answer for this data item, for EVERY sample of the paper. Rejected if any check fails.

        value / sample_name ... -> the main (best) sample
        other_samples           -> every other sample made and measured in THIS paper, each with
                                   sample_name, value (or range), unit, evidence
        Every sample needs its name as written in the paper; quotes that cite other work are rejected.
        """
        value = self._as_value(value)
        range, error = (range or "").strip(), (error or "").strip()
        range_evidence = (range_evidence or "").strip() or evidence
        error_evidence = (error_evidence or "").strip() or evidence
        others_in = other_samples if other_samples is not None else (all_values or [])

        if value is None and not range and not others_in:     # "not reported" is a valid answer
            self.result = {"value": None, "range": "", "error": "", "unit": unit,
                           "sample_name": sample_name, "evidence": evidence, "note": note,
                           "other_samples": [], "warnings": [], "status": "not_reported"}
            return {"saved": True, "status": "not_reported"}

        problems, warnings = [], []
        if value is not None or range:
            p, w = self._check_one(f"main sample '{sample_name}'", value, unit, evidence, sample_name,
                                   range, range_evidence, error, error_evidence)
            problems += p
            warnings += w

        others, seen = [], {_normalize(sample_name)} if sample_name else set()
        for i, o in enumerate(others_in, 1):
            if not isinstance(o, dict):
                problems.append(f"other_samples[{i}] must be an object with sample_name, value and evidence.")
                continue
            name = str(o.get("sample_name") or o.get("sample") or "").strip()
            o_value = self._as_value(o.get("value"))
            o_range = str(o.get("range") or "").strip()
            o_unit = str(o.get("unit") or unit or "").strip()
            o_ev = str(o.get("evidence") or "").strip()
            o_rev = str(o.get("range_evidence") or "").strip() or o_ev
            o_err = str(o.get("error") or "").strip()
            o_eev = str(o.get("error_evidence") or "").strip() or o_ev
            label = f"other sample '{name or i}'"
            if o_value is None and not o_range:
                problems.append(f"{label}: give a value (or a range). Leave out samples with no reported value.")
                continue
            if _normalize(name) in seen:
                problems.append(f"{label}: this sample is listed twice.")
                continue
            seen.add(_normalize(name))
            p, w = self._check_one(label, o_value, o_unit, o_ev, name, o_range, o_rev, o_err, o_eev)
            problems += p
            warnings += w
            others.append({"sample_name": name, "value": o_value, "range": o_range, "error": o_err,
                           "unit": o_unit, "evidence": o_ev, "range_evidence": o_rev if o_range else "",
                           "error_evidence": o_eev if o_err else "",
                           "method": str(o.get("method") or method or ""), "note": str(o.get("note") or ""),
                           "status": "verified" if o_value is not None else "range_only"})

        if problems:
            return {"saved": False, "problems": problems,
                    "instruction": "Fix these problems and call save_result again (you can drop samples whose "
                                   "value you cannot quote), or call save_result with value=null if this data "
                                   "is not reported for any sample of this paper."}

        if value is None and not range and others:     # no main value: promote the first other sample
            first = others.pop(0)
            value, range, error, unit = first["value"], first["range"], first["error"], first["unit"]
            sample_name, evidence = first["sample_name"], first["evidence"]
            range_evidence, error_evidence = first["range_evidence"] or evidence, first["error_evidence"] or evidence
        status = "verified" if value is not None else "range_only"
        self.result = {"value": value, "range": range, "error": error, "unit": unit,
                       "sample_name": sample_name, "method": method, "evidence": evidence,
                       "range_evidence": range_evidence if range else "",
                       "error_evidence": error_evidence if error else "",
                       "other_samples": others, "warnings": warnings, "note": note, "status": status}
        out = {"saved": True, "status": status, "n_samples": 1 + len(others)}
        if warnings:
            out["warnings"] = warnings
        return out

    # ----------------------------------------------------- dispatch by name
    def run(self, name: str, args: dict) -> dict:
        fn = {"search_text": self.search_text, "check_evidence": self.check_evidence,
              "check_plausible": self.check_plausible, "convert_unit": self.convert_unit,
              "save_result": self.save_result}.get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"Wrong arguments for {name}: {e}"}


# ---- Tool descriptions sent to the LLM (OpenAI tool-calling format) ----------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_text",
        "description": "Search the paper for a keyword and return the sentences that contain it. "
                       "Try several keywords: the quantity's name, synonyms, its unit, the method used to "
                       "measure it, and sample names.",
        "parameters": {"type": "object",
                       "properties": {"keyword": {"type": "string"}},
                       "required": ["keyword"]}}},
    {"type": "function", "function": {
        "name": "check_evidence",
        "description": "Check that a quote really appears in the paper text.",
        "parameters": {"type": "object",
                       "properties": {"quote": {"type": "string"}},
                       "required": ["quote"]}}},
    {"type": "function", "function": {
        "name": "check_plausible",
        "description": "Check that a numeric value is in the expected range for the data item "
                       "you are currently extracting.",
        "parameters": {"type": "object",
                       "properties": {"value": {"type": "number"},
                                      "unit": {"type": "string",
                                               "description": "Unit as written in the paper. The value is "
                                                              "converted to the range's unit before checking."}},
                       "required": ["value"]}}},
    {"type": "function", "function": {
        "name": "convert_unit",
        "description": "Convert a value from one unit to another with the Pint library, e.g. 5e-9 m -> nm, "
                       "6 mmol g-1 h-1 -> µmol g-1 h-1, 45 m2/g -> m2 g-1. Use it to compare values reported in "
                       "different units. Do NOT use it to change what you save: save_result must have the value "
                       "and unit exactly as written in the paper (results are standardised afterwards).",
        "parameters": {"type": "object",
                       "properties": {"value": {"type": "number"},
                                      "from_unit": {"type": "string", "description": "e.g. 'mmol g-1 h-1'"},
                                      "to_unit": {"type": "string", "description": "e.g. 'µmol g-1 h-1'"}},
                       "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {
        "name": "save_result",
        "description": "Save the FINAL answer for the current data item, for EVERY sample of this paper: "
                       "the main (best) sample in value/sample_name, all other samples in other_samples. "
                       "The system checks every value, sample name and quote; quotes citing other work "
                       "([12], et al., previously reported) are rejected. If a check fails, fix it and retry. "
                       "Give a specific value if the paper reports one. If it only gives a range or bound, "
                       "set value=null and fill 'range'. Fill 'error' if an uncertainty is reported. "
                       "Use value=null and range='' if the paper does not report this data.",
        "parameters": {"type": "object",
                       "properties": {
                           "value": {"type": ["number", "string", "null"],
                                     "description": "The value for the main sample, exactly as written in the "
                                                    "evidence (a number, or text for non-numeric data), or null."},
                           "unit": {"type": "string", "description": "Unit as written in the paper, e.g. 'eV'."},
                           "evidence": {"type": "string",
                                        "description": "Exact quote from the paper containing the value."},
                           "sample_name": {"type": "string",
                                           "description": "Name of the MAIN (best) sample exactly as written in "
                                                          "the paper, e.g. 'TiO2-500'. Required."},
                           "method": {"type": "string",
                                      "description": "How it was determined, e.g. 'Tauc plot from UV-vis DRS'."},
                           "other_samples": {
                               "type": "array",
                               "description": "EVERY other sample made and measured in THIS paper that has a "
                                              "value for this data item (pristine/reference samples too, e.g. "
                                              "bulk g-C3N4 or P25 if the authors measured them). Each needs "
                                              "its own name and quote; one quote can cover several samples "
                                              "(a table row, or a '... respectively' sentence). Never include "
                                              "values from other papers.",
                               "items": {"type": "object",
                                         "properties": {
                                             "sample_name": {"type": "string",
                                                             "description": "Name exactly as in the paper."},
                                             "value": {"type": ["number", "string", "null"]},
                                             "unit": {"type": "string"},
                                             "evidence": {"type": "string",
                                                          "description": "Exact quote containing this value."},
                                             "range": {"type": "string"},
                                             "error": {"type": "string"}},
                                         "required": ["sample_name", "value", "evidence"]}},
                           "range": {"type": "string",
                                     "description": "ONLY if the paper gives a range or bound instead of (or as "
                                                    "well as) a specific value, copied as written, e.g. "
                                                    "'2.6-2.8' or '>160'. Otherwise ''."},
                           "range_evidence": {"type": "string",
                                              "description": "Exact quote containing the range, if it is not "
                                                             "already in 'evidence'."},
                           "error": {"type": "string",
                                     "description": "Error / uncertainty reported for this data, copied as "
                                                    "written, e.g. '± 0.05' or '0.3 (standard deviation)'. "
                                                    "Otherwise ''."},
                           "error_evidence": {"type": "string",
                                              "description": "Exact quote containing the error, if it is not "
                                                             "already in 'evidence'."},
                           "note": {"type": "string"}},
                       "required": ["value", "evidence", "sample_name"]}}},
]
