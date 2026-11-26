"""
Conversions that need more than units: they need extra data from the paper.

1. Reference electrodes      V vs. Ag/AgCl -> V vs. NHE (fixed offset), V vs. RHE <-> NHE (needs the pH)
2. Catalyst-mass normalising µmol h-1 -> µmol g-1 h-1 (needs the mass of catalyst used)
3. Molar mass                mg g-1 h-1 of H2 -> µmol g-1 h-1 (molar mass from the formula, via periodictable)

When a conversion needs a value it does not have, it raises NeedsValue("pH") / NeedsValue("catalyst mass");
the agent then extracts that value from the paper and tries again (see agent.standardise_results).
"""
import re
from collections.abc import Callable

import periodictable

from .units import UnitError, convert, parse_unit, same_kind, strip_reference, ureg

# ============================================================ 1. reference electrodes
# Potential of each reference electrode vs. NHE at 25 °C, in V. E(vs NHE) = E(vs ref) + offset.
REFERENCE_OFFSETS_V = {
    "NHE": 0.0,
    "SHE": 0.0,
    "Ag/AgCl": 0.197,              # saturated KCl (the usual one); 3 M KCl is 0.210
    "Ag/AgCl (3 M KCl)": 0.210,
    "SCE": 0.241,                  # saturated calomel
    "Hg/HgO": 0.098,               # 1 M NaOH
}
NERNST_SLOPE_V = 0.0591            # V per pH unit at 25 °C: E(vs RHE) = E(vs NHE) + 0.0591 * pH
REFERENCES = (*REFERENCE_OFFSETS_V, "RHE")

_REFERENCE = re.compile(r"vs\.?\s*(?:the\s+)?(RHE|NHE|SHE|SCE|Ag\s*/\s*AgCl|Hg\s*/\s*HgO)"
                        r"(\s*\(?\s*3\s*M\s*KCl)?", re.I)


class NeedsValue(Exception):
    """A conversion needs a value from the paper that has not been extracted yet."""

    def __init__(self, name: str, why: str):
        super().__init__(f"needs the {name}: {why}")
        self.name = name
        self.why = why


def find_reference(*texts: str) -> str | None:
    """The reference electrode named in any of the texts ('V vs. RHE', '... vs. Ag/AgCl'), else None."""
    for text in texts:
        m = _REFERENCE.search(text or "")
        if m:
            ref = re.sub(r"\s+", "", m.group(1)).upper().replace("AGCL", "AgCl").replace("HGO", "HgO")
            ref = ref.replace("AG/", "Ag/").replace("HG/", "Hg/")
            return "Ag/AgCl (3 M KCl)" if ref == "Ag/AgCl" and m.group(2) else ref
    return None


def _offset_v(ref: str, pH: float | None) -> float:
    if ref == "RHE":
        if pH is None:
            raise NeedsValue("pH", "converting to or from the RHE scale depends on the pH (E_RHE = E_NHE + 0.0591 pH)")
        return -NERNST_SLOPE_V * pH
    if ref not in REFERENCE_OFFSETS_V:
        raise UnitError(f"Unknown reference electrode '{ref}'. Known: {', '.join(REFERENCES)}.")
    return REFERENCE_OFFSETS_V[ref]


def potential_shift(from_ref: str, to_ref: str, pH: float | None = None) -> float:
    """Volts to ADD to a potential to move it from one reference scale to another."""
    if from_ref == to_ref:
        return 0.0
    return _offset_v(from_ref, pH) - _offset_v(to_ref, pH)


def convert_potential(value: float, from_unit: str, to_unit: str, pH: float | None = None) -> float:
    """'-0.4 V vs. Ag/AgCl' -> V vs. NHE etc. Both units must name their reference electrode."""
    from_ref, to_ref = find_reference(from_unit), find_reference(to_unit)
    if not from_ref or not to_ref:
        raise UnitError("Both units must name a reference electrode, e.g. 'V vs. RHE' and 'V vs. NHE'.")
    volts = convert(value, strip_reference(from_unit), "V")
    return convert(volts + potential_shift(from_ref, to_ref, pH), "V", strip_reference(to_unit))


# ============================================================ 3. molar mass
def molar_mass(formula: str) -> float:
    """Molar mass in g/mol from a chemical formula ('H2' -> 2.016, 'C3N4' -> 92.06, 'TiO2' -> 79.87)."""
    clean = re.sub(r"^[a-zA-Z]{1,2}-(?=[A-Z])", "", formula.strip())     # 'g-C3N4' -> 'C3N4'
    try:
        return float(periodictable.formula(clean).mass)
    except Exception as e:      # periodictable raises several kinds of parse errors
        raise UnitError(f"Cannot read the formula '{formula}' ({e}).") from e


# ============================================================ 2 + 3. amount-based rates
_AMOUNT = re.compile(r"^\s*([a-zA-Zµμ]+)(?=\s|/|$)")


def split_amount(unit: str) -> tuple[str, str]:
    """Split a rate into the amount of product and the rest: 'mg g-1 h-1' -> ('mg', 'g-1 h-1'),
    'µmol/h' -> ('µmol', '/h'). Pint alone cannot do this, because it would cancel the mg of H2
    against the g of catalyst."""
    m = _AMOUNT.match(unit)
    if not m:
        raise UnitError(f"No amount found at the start of '{unit}'.")
    return m.group(1), unit[m.end():].strip()


def _kind(unit: str) -> str:
    dims = parse_unit(unit).dimensionality
    if dims == ureg.Unit("mol").dimensionality:
        return "substance"
    if dims == ureg.Unit("g").dimensionality:
        return "mass"
    return "other"


def _rest_unit(rest: str) -> str:
    """'/h' -> 'h-1', 'g-1 h-1' stays; '' -> '' (dimensionless)."""
    rest = rest.strip()
    return f"1 {rest}".replace("1 /", "1/") if rest.startswith("/") else rest


def amount_rate_converter(from_unit: str, to_unit: str, species: str | None = None,
                          catalyst_mass: tuple[float, str] | None = None) -> tuple[Callable[[float], float], str]:
    """A function converting rates like 'mg h-1' -> 'µmol g-1 h-1', and a note saying what it did.

    - amount of product: mass <-> moles with the molar mass of `species` (e.g. 'H2')
    - normalisation: if the target is per gram of catalyst and the value is not, divide by `catalyst_mass`
    Raises NeedsValue('catalyst mass') if the mass is needed and missing, UnitError if impossible.
    """
    amount_from, rest_from = split_amount(from_unit)
    amount_to, rest_to = split_amount(to_unit)
    kinds = (_kind(amount_from), _kind(amount_to))
    if "other" in kinds:
        raise UnitError(f"'{amount_from}' or '{amount_to}' is not an amount of product (mass or moles).")
    notes = []

    # amount of product: same kind -> plain; mass <-> moles -> molar mass
    factor = convert(1.0, amount_from, amount_to) if kinds[0] == kinds[1] else None
    if factor is None:
        if not species:
            raise UnitError(f"Converting {amount_from} to {amount_to} needs the molar mass of the product, "
                            f"but no 'species' is set for this parameter in the dictionary.")
        mw = molar_mass(species)
        grams_per = convert(1.0, amount_from, "g") if kinds[0] == "mass" else None
        if grams_per is not None:                               # mass -> moles
            factor = convert(grams_per / mw, "mol", amount_to)
        else:                                                   # moles -> mass
            factor = convert(convert(1.0, amount_from, "mol") * mw, "g", amount_to)
        notes.append(f"molar mass of {species} = {mw:g} g/mol")

    # the rest (per g per h ...): same kind -> plain; missing a per-mass -> divide by the catalyst mass
    rf, rt = _rest_unit(rest_from) or "dimensionless", _rest_unit(rest_to) or "dimensionless"
    if same_kind(rf, rt):
        rest_factor, mass_g = convert(1.0, rf, rt), None
    elif parse_unit(rf).dimensionality / ureg.Unit("g").dimensionality == parse_unit(rt).dimensionality:
        if catalyst_mass is None:
            raise NeedsValue("catalyst mass", f"'{from_unit}' is not per gram of catalyst but '{to_unit}' is")
        mass_g = convert(catalyst_mass[0], catalyst_mass[1], "g")
        if mass_g <= 0:
            raise UnitError("The catalyst mass must be positive.")
        rest_factor = convert(1.0, f"{rf} g-1", rt)
        notes.append(f"divided by catalyst mass {catalyst_mass[0]:g} {catalyst_mass[1]}")
    else:
        raise UnitError(f"Cannot convert '{from_unit}' to '{to_unit}'.")

    total = factor * rest_factor / (mass_g if mass_g else 1.0)
    return (lambda x: x * total), "; ".join(notes)
