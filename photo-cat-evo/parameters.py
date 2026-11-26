"""
The parameter dictionary: experimental quantities the agent knows about (band gap, BET surface area, ...).

The entries live in parameters.json so you can add to them without touching the code.
Each entry is keyed by its name and can have:
    aliases          other names for it in prompts/papers ("bandgap", "Eg")
    unit             the usual unit ("eV"). After extraction every result is converted to this unit
                     (with Pint, see units.py) and shown in the standard_value / standard_unit columns
    standard_unit    optional: convert to this unit instead of "unit" (e.g. unit "nm", standard_unit "m")
    plausible_range  [min, max] in that unit; values outside it are rejected (only checked when the
                     paper uses the same unit)
    search_terms     keywords the agent should try when searching the paper
    methods          how it is usually measured
    description      notes for the agent: what it is, how papers write it, what to watch out for

Add an entry by editing parameters.json, or from the command line:
    python parameters.py list
    python parameters.py show "band gap"
    python parameters.py add "zeta potential" --unit mV --range -100 100 \\
        --aliases "surface charge" --search "zeta potential,mV" \\
        --methods "Electrophoretic light scattering" --description "Surface charge of particles in suspension."
"""
import argparse
import json
import re

from config import PARAMETERS_FILE
from units import UnitError, parse_unit

FIELDS = ("aliases", "unit", "standard_unit", "plausible_range", "search_terms", "methods", "description",
          "allowed_values", "default_if_missing")


# ============================================================ load / save
def load_parameters(path: str = PARAMETERS_FILE) -> dict:
    """Read the dictionary. A missing file means an empty dictionary (the agent still works)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON (line {e.lineno}): {e.msg}. "
                         f"Check for a missing comma or quote near that line.") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object: {{\"name\": {{...}}, ...}}")
    return data


def add_parameter(name: str, entry: dict, path: str = PARAMETERS_FILE, overwrite: bool = False) -> dict:
    """Add (or with overwrite=True replace) one entry and save. Returns the updated dictionary."""
    name = name.strip()
    if not name:
        raise ValueError("The parameter needs a name.")
    params = load_parameters(path)
    existing = find_parameter(name, params)
    if existing and not overwrite:
        raise ValueError(f"'{name}' is already in the dictionary as '{existing[0]}'. "
                         f"Use --overwrite to replace it, or edit {path}.")
    for field in ("unit", "standard_unit"):
        if entry.get(field):
            try:
                parse_unit(entry[field])
            except UnitError as e:
                raise ValueError(f"{field}: {e} Write it like 'm2 g-1', 'µmol g-1 h-1' or 'nm'.") from e
    rng = entry.get("plausible_range")
    if rng is not None and (len(rng) != 2 or float(rng[0]) > float(rng[1])):
        raise ValueError("plausible_range must be [min, max] with min <= max.")
    clean = {k: entry[k] for k in FIELDS if entry.get(k) not in (None, "", [])}
    updated = {**params, name: clean}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return updated


# ============================================================ lookup
def _words(s: str) -> str:
    """'H2-evolution  Rate' -> 'h2 evolution rate' (lowercase, punctuation to single spaces)."""
    return " ".join(re.sub(r"[^0-9a-zµ%]+", " ", s.lower()).split())


def find_parameter(item_name: str, params: dict | None = None) -> tuple[str, dict] | None:
    """Match a data item from the prompt to a dictionary entry. Returns (name, entry) or None.

    1. exact match on the name or an alias ("bandgap" -> "band gap")
    2. otherwise the longest name/alias that appears as whole words in the item
       ("H2 evolution rate under visible light" -> "H2 evolution rate").
       Very short aliases (e.g. "Eg", "CB") are only used for exact matches.
    """
    params = load_parameters() if params is None else params
    item = _words(item_name)
    candidates = [(_words(alias), name)
                  for name, entry in params.items()
                  for alias in [name, *entry.get("aliases", [])]]
    for alias, name in candidates:
        if alias == item:
            return name, params[name]
    contained = [(len(alias), name) for alias, name in candidates
                 if len(alias) >= 4 and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", item)]
    if contained:
        name = max(contained)[1]
        return name, params[name]
    return None


def standard_unit_for(entry: dict | None) -> str:
    """The unit results for this parameter are converted to ('' if the entry has none)."""
    entry = entry or {}
    return (entry.get("standard_unit") or entry.get("unit") or "").strip()


# ============================================================ units
def _unit_parts(unit: str) -> set[tuple[str, int]]:
    """'µmol g-1 h-1', 'umol/g/h' and 'μmol·h⁻¹·g⁻¹' all -> {('µmol',1), ('g',-1), ('h',-1)}."""
    u = unit.lower().replace("μ", "µ").replace("−", "-").replace("⁻", "-")
    u = u.translate(str.maketrans("¹²³", "123")).replace("·", " ").replace("*", " ")
    u = re.sub(r"\bu(?=mol|a\b|m\b|l\b|g\b)", "µ", u)          # 'umol' -> 'µmol'
    u = re.sub(r"/\s*([a-zµ%]+)(\d*)", lambda m: f" {m.group(1)}-{m.group(2) or '1'}", u)
    parts = re.findall(r"([a-zµ%]+)\s*(-?\d*)", u)
    return {(base, int(exp) if exp not in ("", "-") else 1) for base, exp in parts}


def units_match(a: str, b: str) -> bool:
    """True if two unit strings mean the same unit (ignores spacing, order, '/' vs '-1')."""
    return bool(a.strip()) and bool(b.strip()) and _unit_parts(a) == _unit_parts(b)


# ============================================================ for the agent
def describe_for_planner(params: dict) -> str:
    """One line per entry, so the planner can use the dictionary names."""
    lines = []
    for name, entry in params.items():
        aliases = ", ".join(entry.get("aliases", []))
        unit = f" [{entry['unit']}]" if entry.get("unit") else ""
        lines.append(f"- {name}{unit}" + (f" (also called: {aliases})" if aliases else ""))
    return "\n".join(lines)


def describe_for_extractor(name: str, entry: dict) -> str:
    """The notes the extractor gets for one data item."""
    lines = [f"Notes on '{name}' from the parameter dictionary:"]
    if entry.get("description"):
        lines.append(f"- What it is: {entry['description']}")
    if entry.get("unit"):
        lines.append(f"- Usual unit: {entry['unit']} (still report the unit exactly as the paper writes it; "
                     f"results are converted to {standard_unit_for(entry)} afterwards)")
    if entry.get("aliases"):
        lines.append(f"- Other names: {', '.join(entry['aliases'])}")
    if entry.get("methods"):
        lines.append(f"- Usually measured by: {'; '.join(entry['methods'])}")
    if entry.get("search_terms"):
        lines.append(f"- Good search terms: {', '.join(entry['search_terms'])}")
    if entry.get("plausible_range"):
        lo, hi = entry["plausible_range"]
        lines.append(f"- Expected range: {lo}-{hi} {entry.get('unit') or ''}".rstrip())
    if entry.get("allowed_values"):
        lines.append(f"- Usual answers (text, not a number): {', '.join(entry['allowed_values'])}")
    if entry.get("default_if_missing"):
        lines.append(f"- If the paper does not state it, save '{entry['default_if_missing']}' and say "
                     f"'not stated, default' in note (the evidence can be the sentence describing the procedure)")
    return "\n".join(lines)


# ============================================================ command line
def _split(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="View or add to the parameter dictionary.",
                                     epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default=PARAMETERS_FILE, help=f"Dictionary file (default {PARAMETERS_FILE})")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List all parameters")
    show = sub.add_parser("show", help="Show one parameter")
    show.add_argument("name")
    add = sub.add_parser("add", help="Add a parameter")
    add.add_argument("name")
    add.add_argument("--unit", default="", help="Usual unit; results are converted to it")
    add.add_argument("--standard-unit", default="", help="Convert results to this unit instead of --unit")
    add.add_argument("--aliases", help="Comma-separated other names")
    add.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"), help="Plausible range")
    add.add_argument("--search", help="Comma-separated search terms")
    add.add_argument("--methods", help="Comma-separated measurement methods")
    add.add_argument("--description", default="")
    add.add_argument("--overwrite", action="store_true", help="Replace an existing entry")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        params = load_parameters(args.file)
        print(describe_for_planner(params) or "(empty)")
        print(f"\n{len(params)} parameter(s) in {args.file}")
    elif args.cmd == "show":
        match = find_parameter(args.name, load_parameters(args.file))
        if not match:
            raise SystemExit(f"'{args.name}' is not in the dictionary. Add it with: python parameters.py add ...")
        print(describe_for_extractor(*match))
    else:
        entry = {"aliases": _split(args.aliases), "unit": args.unit.strip(),
                 "standard_unit": args.standard_unit.strip(),
                 "plausible_range": list(args.range) if args.range else None,
                 "search_terms": _split(args.search), "methods": _split(args.methods),
                 "description": args.description.strip()}
        try:
            add_parameter(args.name, entry, args.file, args.overwrite)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"Added '{args.name}' to {args.file}.")


if __name__ == "__main__":
    main()
