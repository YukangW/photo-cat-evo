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

from parameters import find_parameter, load_parameters
from units import UnitError, convert


def _normalize(s: str) -> str:
    """Make text comparable: lowercase, unify dashes/minus/spaces, drop extra whitespace."""
    s = s.lower()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace(" ", " ").replace(" ", " ")
    # remove ALL whitespace: PDFs often split formulas ("C3 N4 -VI") or lines unpredictably,
    # while the model usually writes them without spaces ("C3N4-VI")
    return re.sub(r"\s+", "", s)


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
        """Return sentences that contain the keyword (case-insensitive)."""
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", self.text))
        hits = [s.strip() for s in sentences if keyword.lower() in s.lower()]
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

    def save_result(self, value=None, unit: str = "", evidence: str = "", sample_name: str = "",
                    method: str = "", all_values: list | None = None, note: str = "",
                    range: str = "", range_evidence: str = "",
                    error: str = "", error_evidence: str = "") -> dict:
        """Final answer for this data item. Rejected (agent must fix it) if the checks fail.

        value  -> a specific value reported in the paper (preferred)
        range  -> fallback when only a range or bound is given, e.g. "2.6-2.8" or ">160"
        error  -> uncertainty reported with the value or range, e.g. "± 0.05"
        range_evidence / error_evidence default to evidence when they come from the same quote.
        """
        if isinstance(value, str) and not value.strip():
            value = None
        if isinstance(value, str):     # the model sometimes sends a number as a string ("3.05")
            try:
                value = float(value.strip())
            except ValueError:
                pass
        range, error = (range or "").strip(), (error or "").strip()
        range_evidence = (range_evidence or "").strip() or evidence
        error_evidence = (error_evidence or "").strip() or evidence

        if value is None and not range:          # "not reported" is a valid answer
            self.result = {"value": None, "range": "", "error": "", "unit": unit,
                           "sample_name": sample_name, "evidence": evidence, "note": note,
                           "status": "not_reported"}
            return {"saved": True, "status": "not_reported"}

        problems = []
        if value is not None:
            ev = self.check_evidence(evidence)
            if not ev["found"]:
                problems.append(ev["reason"])
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not _number_in(value, evidence):
                    problems.append(f"The value {value} does not appear in the evidence quote. "
                                    f"Report the number exactly as written in the quote.")
                pl = self.check_plausible(value, unit)
                if not pl["plausible"]:
                    problems.append(pl["reason"])
            elif _normalize(str(value)) not in _normalize(evidence):   # text answer, e.g. "anatase"
                problems.append(f"The text value '{value}' does not appear in the evidence quote.")
        if range:
            problems += self._check_extra("range", range, range_evidence)
        if error:
            problems += self._check_extra("error", error, error_evidence)

        if problems:
            return {"saved": False, "problems": problems,
                    "instruction": "Fix these problems and call save_result again, "
                                   "or call save_result with value=null if this data is not reported."}

        status = "verified" if value is not None else "range_only"
        self.result = {"value": value, "range": range, "error": error, "unit": unit,
                       "sample_name": sample_name, "method": method, "evidence": evidence,
                       "range_evidence": range_evidence if range else "",
                       "error_evidence": error_evidence if error else "",
                       "all_values": all_values or [], "note": note, "status": status}
        return {"saved": True, "status": status}

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
        "description": "Save the FINAL answer for the current data item. The system checks the evidence "
                       "and the value; if a check fails the result is rejected and you must fix it. "
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
                                           "description": "Sample name as written in the paper, e.g. 'TiO2-500'."},
                           "method": {"type": "string",
                                      "description": "How it was determined, e.g. 'Tauc plot from UV-vis DRS'."},
                           "all_values": {"type": "array",
                                          "description": "Values for other samples, "
                                                         "e.g. [{'sample':'P25','value':3.2}].",
                                          "items": {"type": "object"}},
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
                       "required": ["value", "evidence"]}}},
]
