"""
Step 3 of the agent: convert one extracted result to the standard unit for its kind of measurement.

The target comes from the parameter dictionary entry:
    unit / standard_unit   the standard unit                      e.g. "µmol g-1 h-1"
    contexts               Pint contexts allowed                  e.g. ["spectroscopy"] for eV <-> nm
    reference              standard reference electrode           e.g. "NHE"
    species                product, for its molar mass            e.g. "H2"  (mg -> µmol)
Items not in the dictionary go to the SI base unit (dimensionless values and % stay as they are).

The conversion is tried in this order:
    1. plain unit change (Pint)                    mmol -> µmol, Å -> nm
    2. a Pint context from the entry               455 nm -> 2.72 eV
    3. amount-based rates (conversions.py)         mg H2 -> µmol H2, µmol h-1 -> µmol g-1 h-1 (catalyst mass)
  + a reference electrode shift if the entry has a reference (needs the pH for RHE)

If a value from the paper is missing (pH, catalyst mass), the result says so in "needs" and the agent
extracts it and calls standardise again with it in `supporting`.
"""
from .conversions import NeedsValue, amount_rate_converter, find_reference, potential_shift
from .parameters import standard_unit_for
from .units import CONTEXTS, UnitError, convert, map_error, map_range, needs_context, parse_unit, si_unit, \
    strip_reference

COLUMNS = ("standard_value", "standard_range", "standard_error", "standard_unit", "conversion_note")


def _empty() -> dict:
    return {"standard_value": None, "standard_range": "", "standard_error": "", "standard_unit": "",
            "conversion_note": "", "needs": []}


def _converter(result: dict, unit: str, target: str, entry: dict, supporting: dict):
    """(function value -> standard value, display unit, list of notes). Raises UnitError / NeedsValue."""
    unit_part, target_part = strip_reference(unit), strip_reference(target)
    notes = []

    # reference electrode (potentials)
    to_ref = entry.get("reference")
    shift = 0.0
    if to_ref:
        from_ref = find_reference(unit, result.get("evidence", ""), result.get("note", ""),
                                  result.get("method", ""))
        if not from_ref:
            raise UnitError(f"The reference electrode is not stated, so it could not be put on the {to_ref} scale.")
        shift = potential_shift(from_ref, to_ref, supporting.get("pH"))
        if from_ref != to_ref:
            pH = f" at pH {supporting['pH']:g}" if "RHE" in (from_ref, to_ref) else ""
            notes.append(f"{from_ref} -> {to_ref} scale ({shift:+.4g} V{pH})")
        target_display = f"{target_part} vs. {to_ref}"
    else:
        target_display = target_part

    # the unit itself
    contexts = entry.get("contexts", [])
    try:
        convert(1.0, unit_part, target_part)
        scale = lambda x: convert(x, unit_part, target_part)  # noqa: E731
    except UnitError as plain_error:
        ctx = needs_context(unit_part, target_part, contexts)
        if ctx:
            scale = lambda x: convert(x, unit_part, target_part, [ctx])  # noqa: E731
            notes.append(f"{ctx} context, {CONTEXTS[ctx]}")
        else:
            try:
                scale, how = amount_rate_converter(unit_part, target_part, entry.get("species"),
                                                   supporting.get("catalyst mass"))
                notes.append(how)
            except UnitError:
                raise plain_error from None

    if shift:   # convert the unit first, then shift on the volt scale
        fn = lambda x: convert(convert(scale(x), target_part, "V") + shift, "V", target_part)  # noqa: E731
    else:
        fn = scale
    return fn, target_display, [n for n in notes if n]


def standardise(result: dict, entry: dict | None = None, supporting: dict | None = None) -> dict:
    """The standard_* columns for one extracted result, plus 'needs': values to extract first.

    entry       the parameter dictionary entry for this item (None if it is not in the dictionary)
    supporting  values extracted for conversions: {"pH": 7.0, "catalyst mass": (50.0, "mg")}
    """
    entry, supporting = entry or {}, supporting or {}
    out = _empty()
    if result.get("status") not in ("verified", "range_only"):
        return out
    value, unit = result.get("value"), (result.get("unit") or "").strip()
    standard_unit = standard_unit_for(entry)
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
            target, how = strip_reference(unit), "dimensionless, kept as reported"
        else:
            target, how = si_unit(unit), "SI base unit, as it is not in the parameter dictionary"
        fn, display, notes = _converter(result, unit, target, entry, supporting)
        converted = {
            "standard_value": fn(value) if value is not None else None,
            "standard_range": map_range(result["range"], fn) if result.get("range") else "",
            "standard_error": map_error(result["error"], fn, at=value) if result.get("error") else "",
        }
    except NeedsValue as e:
        return {**out, "needs": [e.name], "conversion_note": f"Not converted: {e}."}
    except UnitError as e:
        return {**out, "conversion_note": str(e)}

    if parse_unit(strip_reference(unit)) == parse_unit(strip_reference(target)) and not notes:
        note = f"Already in {display}."
    else:
        note = f"Converted from {unit} to {display} ({'; '.join([how, *notes])})."
    if not entry.get("reference"):
        ref = find_reference(unit)
        if ref:
            note += f" Reference electrode: vs. {ref}."
    return {**out, **converted, "standard_unit": display, "conversion_note": note}
