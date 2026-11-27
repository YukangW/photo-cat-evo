"""
Unit conversion with Pint (https://pint.readthedocs.io).

Papers write units in many ways ("m2 g−1", "m²/g", "µmol h-1 g-1", "umol/g/h"); parse_unit() turns them
into something Pint understands, and the functions below convert values, errors and ranges.

    convert(5e-9, "m", "nm")                                  -> 5.0
    convert(6.0, "mmol g−1 h−1", "µmol g-1 h-1")              -> 6000.0
    convert(2.72, "eV", "nm", contexts=["spectroscopy"])      -> 455.8   (E = hc/λ)

Plain conversions only work between units of the same kind (length -> length). Pint "contexts" add the
physics that links different kinds:
    spectroscopy   energy <-> wavelength <-> wavenumber <-> frequency   (E = hc/λ = hν)
    boltzmann      energy <-> temperature                               (E = k_B T)
Contexts are only used when they are asked for, so a wavelength is never turned into an energy by accident.

Conversions that need data from the paper (reference electrodes + pH, catalyst mass, molar mass)
are in conversions.py; standardise.py puts it all together for one extracted result.
"""
import re
from collections.abc import Callable, Iterable
from threading import RLock

import pint

ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"           # short, pretty: "µmol/g/h", "m²/kg"
# Pint keeps active conversion contexts on the shared registry.
_ureg_lock = RLock()

# Custom context: band potentials are often written "1.94 eV vs. NHE" = energy per electron, i.e. 1.94 V.
_electron = pint.Context("electron")
_electron.add_transformation("[energy]", "[electric_potential]", lambda reg, x: x / reg.elementary_charge)
_electron.add_transformation("[electric_potential]", "[energy]", lambda reg, x: x * reg.elementary_charge)
ureg.add_context(_electron)

# Contexts the agent may use. Each one is a physical relation, noted in the results when it is used.
CONTEXTS = {"spectroscopy": "E = hc/λ", "boltzmann": "E = k_B·T", "electron": "E = e·V, energy per electron"}

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
# A number in a range/error string. A leading "-" counts as a minus sign only at the start or after
# a space/bracket/comparison sign, so "2.6-2.8" is 2.6 and 2.8 but "-0.5 to -0.3" keeps both minus signs.
_NUMBER = re.compile(r"(?:(?<=^)|(?<=[\s(<>~≈=,±]))-?\d+(?:\.\d+)?(?:[eE]-?\d+)?|\d+(?:\.\d+)?(?:[eE]-?\d+)?")
_FLIP = str.maketrans({"<": ">", ">": "<", "≤": "≥", "≥": "≤"})


class UnitError(ValueError):
    """A unit that cannot be read, or two units that cannot be converted into each other."""


# ============================================================ reading units
def parse_unit(unit: str) -> pint.Unit:
    """Read a unit as written in a paper. Raises UnitError if Pint cannot make sense of it."""
    raw = unit
    u = strip_reference(unit or "")                         # "V vs. NHE" -> "V" (see conversions.py)
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
        with _ureg_lock:
            return ureg.dimensionless
    expr = " * ".join(f"{base}**{exp}" if exp else base for base, exp in tokens)
    try:
        with _ureg_lock:
            return ureg.Unit(expr)
    except (pint.UndefinedUnitError, pint.DefinitionSyntaxError, AttributeError, ValueError, TypeError) as e:
        raise UnitError(f"Cannot read the unit '{raw}' ({e}).") from e


def strip_reference(unit: str) -> str:
    """'V vs. RHE' -> 'V'. The reference electrode is not part of the unit."""
    return re.split(r"\s+vs\.?\s", unit.strip())[0].strip()


def same_kind(a: str, b: str) -> bool:
    """Do two units measure the same kind of thing (plain conversion possible)?"""
    with _ureg_lock:
        return parse_unit(a).dimensionality == parse_unit(b).dimensionality


def si_unit(unit: str) -> str:
    """The SI base unit for the same kind of quantity: 'ns' -> 's', 'm2 g-1' -> 'm²/kg'."""
    with _ureg_lock:
        return f"{ureg.Quantity(1, parse_unit(unit)).to_base_units().units}"


# ============================================================ converting one value
def _round(x: float) -> float:
    """Drop floating-point noise: 6000.000000000001 -> 6000.0 (keeps 10 significant figures)."""
    return float(f"{x:.10g}")


def _check_contexts(contexts: Iterable[str]) -> tuple[str, ...]:
    contexts = tuple(contexts or ())
    unknown = [c for c in contexts if c not in CONTEXTS]
    if unknown:
        raise UnitError(f"Unknown context(s) {unknown}. Available: {', '.join(CONTEXTS)}.")
    return contexts


def convert(value: float, from_unit: str, to_unit: str, contexts: Iterable[str] = ()) -> float:
    """Convert a value between two units written the way papers write them.
    A plain conversion is tried first; the contexts (e.g. ["spectroscopy"]) only if that is impossible."""
    contexts = _check_contexts(contexts)
    with _ureg_lock:
        q = ureg.Quantity(float(value), parse_unit(from_unit))
        target = parse_unit(to_unit)
        try:
            return _round(q.to(target).magnitude)
        except pint.DimensionalityError as e:
            if not contexts:
                raise UnitError(f"Cannot convert '{from_unit}' to '{to_unit}': they measure different things "
                                f"({e.dim1} vs {e.dim2}).") from e
        try:
            with ureg.context(*contexts):
                return _round(q.to(target).magnitude)
        except pint.DimensionalityError as e:
            raise UnitError(f"Cannot convert '{from_unit}' to '{to_unit}', even with the "
                            f"{'/'.join(contexts)} context.") from e
        except ZeroDivisionError as e:
            raise UnitError(f"{value} {from_unit} has no equivalent in {to_unit} (division by zero).") from e


def needs_context(from_unit: str, to_unit: str, contexts: Iterable[str] = ()) -> str | None:
    """Which of the given contexts a conversion uses, or None if it is a plain conversion."""
    if same_kind(from_unit, to_unit):
        return None
    for c in _check_contexts(contexts):
        try:
            convert(1.0, from_unit, to_unit, [c])
            return c
        except UnitError:
            continue
    return None


# ============================================================ converting ranges and errors
Fn = Callable[[float], float]


def is_affine(fn: Fn) -> bool:
    """True for conversions of the form a*x + b (most unit changes); False for e.g. eV -> nm (1/x)."""
    a, b, c = fn(1.0), fn(2.0), fn(3.0)
    return abs((c - b) - (b - a)) <= 1e-9 * max(1.0, abs(a), abs(b), abs(c))


def is_decreasing(fn: Fn, at: float = 1.0) -> bool:
    """Does a bigger input give a smaller output (e.g. a higher energy is a shorter wavelength)?"""
    at = abs(at) or 1.0
    return fn(at * 1.01) < fn(at)


def map_range(text: str, fn: Fn) -> str:
    """Convert every number in a range string, keeping the rest ('>160', '2.6-2.8', '-0.5 to -0.3').
    If the conversion reverses the order (eV -> nm), '<'/'>' are swapped and 'a-b' becomes 'b-a'."""
    numbers = [float(m.group()) for m in _NUMBER.finditer(text)]
    if not numbers:
        return text
    converted = [f"{_round(fn(n)):g}" for n in numbers]
    if is_decreasing(fn, numbers[0]):
        text = text.translate(_FLIP)
        if len(converted) == 2:
            converted.reverse()
    pieces = iter(converted)
    return _NUMBER.sub(lambda m: next(pieces), text)


def map_error(text: str, fn: Fn, at: float | None = None) -> str:
    """Convert the numbers in an error string ('± 0.02'). An error is a difference, so offsets
    (°C -> K, a reference electrode shift) do not apply. For non-linear conversions (eV -> nm) the
    error depends on the value: it is propagated as |f(x+e) - f(x-e)| / 2 around the value x."""
    affine = is_affine(fn)
    if not affine and at is None:
        raise UnitError("The error cannot be converted without its value (the conversion is not linear).")

    def one(e: float) -> str:
        if affine:
            return f"{_round(abs(fn(1.0 + e) - fn(1.0))):g}"
        return f"{_round(abs(fn(at + e) - fn(at - e)) / 2):g}"

    return _NUMBER.sub(lambda m: one(abs(float(m.group()))), text)


# ---- simple helpers kept for convenience (and older code/tests) ----------------------------------
def convert_error(error: float, from_unit: str, to_unit: str, contexts: Iterable[str] = (),
                  at: float | None = None) -> float:
    """Convert one uncertainty value (see map_error)."""
    return float(map_error(str(abs(error)), lambda x: convert(x, from_unit, to_unit, contexts), at))


def convert_text(text: str, from_unit: str, to_unit: str, as_error: bool = False,
                 contexts: Iterable[str] = (), at: float | None = None) -> str:
    """Convert the numbers in a range ('>160') or, with as_error=True, an error ('± 0.02') string."""
    fn = lambda x: convert(x, from_unit, to_unit, contexts)  # noqa: E731
    return map_error(text, fn, at) if as_error else map_range(text, fn)
