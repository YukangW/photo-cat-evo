"""
Unit conversion with Pint (https://pint.readthedocs.io).

Papers write units in many ways ("m2 g−1", "m²/g", "µmol h-1 g-1", "umol/g/h"); parse_unit() turns them
into something Pint understands, and the functions below convert values, errors and ranges.

    convert(5e-9, "m", "nm")                        -> 5.0
    convert(6.0, "mmol g−1 h−1", "µmol g-1 h-1")    -> 6000.0
    standardise(result, "µmol g-1 h-1")             -> the standard_* columns for one extracted result
"""
import re

import pint

ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"           # short, pretty: "µmol/g/h", "m²/kg"

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
# A number in a range/error string. A leading "-" counts as a minus sign only at the start or after
# a space/bracket/comparison sign, so "2.6-2.8" is 2.6 and 2.8 but "-0.5 to -0.3" keeps both minus signs.
_NUMBER = re.compile(r"(?:(?<=^)|(?<=[\s(<>~≈=,±]))-?\d+(?:\.\d+)?(?:[eE]-?\d+)?|\d+(?:\.\d+)?(?:[eE]-?\d+)?")


class UnitError(ValueError):
    """A unit that cannot be read, or two units that cannot be converted into each other."""


def parse_unit(unit: str) -> pint.Unit:
    """Read a unit as written in a paper. Raises UnitError if Pint cannot make sense of it."""
    raw = unit
    u = (unit or "").strip()
    u = re.split(r"\s+vs\.?\s", u)[0]                       # "V vs. NHE" -> "V" (reference kept in the note)
    if re.search(r"[A-Za-z]\.", u) or re.search(r"\barb", u, re.I):
        raise UnitError(f"'{raw}' is an arbitrary/abbreviated unit (e.g. a.u.) and cannot be converted.")
    u = u.replace("μ", "µ").replace("−", "-").replace("–", "-").replace("⁻", "-").translate(_SUPERSCRIPTS)
    u = u.replace("°C", "degC").replace("℃", "degC").replace("·", " ").replace("⋅", " ").replace("*", " ")
    u = u.replace("%", " percent ").replace("(", " ").replace(")", " ")
    u = re.sub(r"/\s*([A-Za-zµÅ]+)(\d*)", lambda m: f" {m.group(1)}-{m.group(2) or '1'}", u)  # "/g" -> " g-1"
    tokens = re.findall(r"([A-Za-zµÅ]+)\s*(-?\d+)?", u)
    if not tokens:
        if u.strip():
            raise UnitError(f"Cannot read the unit '{raw}'.")
        return ureg.dimensionless
    expr = " * ".join(f"{base}**{exp}" if exp else base for base, exp in tokens)
    try:
        return ureg.Unit(expr)
    except (pint.UndefinedUnitError, pint.DefinitionSyntaxError, AttributeError, ValueError, TypeError) as e:
        raise UnitError(f"Cannot read the unit '{raw}' ({e}).") from e


def _round(x: float) -> float:
    """Drop floating-point noise: 6000.000000000001 -> 6000.0 (keeps 10 significant figures)."""
    return float(f"{x:.10g}")


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between two units written the way papers write them."""
    try:
        q = ureg.Quantity(float(value), parse_unit(from_unit)).to(parse_unit(to_unit))
    except pint.DimensionalityError as e:
        raise UnitError(f"Cannot convert '{from_unit}' to '{to_unit}': they measure different things "
                        f"({e.dim1} vs {e.dim2}).") from e
    return _round(q.magnitude)


def convert_error(error: float, from_unit: str, to_unit: str) -> float:
    """Convert an uncertainty. It is a difference, so offsets (e.g. °C -> K) do not apply."""
    return _round(convert(error, from_unit, to_unit) - convert(0, from_unit, to_unit))


def convert_text(text: str, from_unit: str, to_unit: str, as_error: bool = False) -> str:
    """Convert every number inside a range/error string, keeping the rest: '>160' -> '>160', '± 0.02' -> ..."""
    fn = convert_error if as_error else convert
    return _NUMBER.sub(lambda m: f"{fn(float(m.group()), from_unit, to_unit):g}", text)


def si_unit(unit: str) -> str:
    """The SI base unit for the same kind of quantity: 'ns' -> 's', 'm2 g-1' -> 'm²/kg'."""
    return f"{ureg.Quantity(1, parse_unit(unit)).to_base_units().units}"


def standardise(result: dict, standard_unit: str | None) -> dict:
    """The standardised columns for one extracted result.

    standard_unit is the dictionary's unit for this parameter. If there is none (the item is not in the
    dictionary) the SI base unit is used instead, except for dimensionless values, which are left as they are.
    Returns {standard_value, standard_range, standard_error, standard_unit, conversion_note}.
    """
    out = {"standard_value": None, "standard_range": "", "standard_error": "", "standard_unit": "",
           "conversion_note": ""}
    if result.get("status") not in ("verified", "range_only"):
        return out
    value, unit = result.get("value"), (result.get("unit") or "").strip()
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return {**out, "conversion_note": "Text value, nothing to convert."}
    if not unit:
        if standard_unit:   # the dictionary expects a unit, so a missing one cannot be assumed
            return {**out, "conversion_note": f"No unit reported, so it was not converted to {standard_unit}."}
        return {**out, "standard_value": value, "standard_range": result.get("range") or "",
                "standard_error": result.get("error") or "",
                "conversion_note": "No unit reported; kept as a dimensionless value."}

    try:
        if standard_unit:
            target, how = standard_unit, "dictionary standard unit"
        elif parse_unit(unit).dimensionless:
            target, how = unit, "dimensionless, kept as reported"
        else:
            target, how = si_unit(unit), "SI base unit, as it is not in the parameter dictionary"
        converted = {
            "standard_value": convert(value, unit, target) if value is not None else None,
            "standard_range": convert_text(result["range"], unit, target) if result.get("range") else "",
            "standard_error": convert_text(result["error"], unit, target, as_error=True)
            if result.get("error") else "",
        }
    except UnitError as e:
        return {**out, "conversion_note": str(e)}

    same = parse_unit(unit) == parse_unit(target)
    note = f"Already in {target}." if same else f"Converted from {unit} to {target} ({how})."
    reference = re.search(r"\s+(vs\.?\s.*)$", unit)
    if reference:
        note += f" Reference electrode: {reference.group(1)}."
    return {**out, **converted, "standard_unit": target, "conversion_note": note}
