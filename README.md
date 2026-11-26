# PhotoCatEvo
Robotics and AI for Materials Chemistry 2026 Winter School -- Project 2: Agentic AI for Knowledge Extraction

Group members: Alex Brown (University of Liverpool), Fiona Gordon (Heriot-Watt University), Ihsan Budi (Edinburgh Napier University), Jack Green (University of Liverpool), Léa Gabay (IST), Misbah Ghafoor (Procter & Gamble), Tom Jackson (University of Liverpool), Yukang Wu (Imperial College London)

## Paper data-extraction agent

An agent that reads one paper (PDF) and extracts the data you ask for in a prompt, checking each answer with tools.

If the prompt asks for several pieces of data, the agent first splits it into separate items, then runs
**once per item, in order**. It finishes and verifies one item before starting the next, and each run is
told what the earlier runs found (e.g. which sample is the main one).

```
prompt ──► plan: ["band gap", "BET surface area", "H2 evolution rate"]
              │
              ├─► run 1: search → check → save_result("band gap")          ✔ verified
              ├─► run 2: search → check → save_result("BET surface area")  ✔ verified
              └─► run 3: search → check → save_result("H2 evolution rate") ✔ / not_reported / failed
```

## Files

| File or directory | What it does |
|---|---|
| `pyproject.toml` and `uv.lock` | Project metadata, the `PhotoCatEvo` command, and locked dependencies |
| `src/PhotoCatEvo/agent.py` | Plans requests and runs the extraction loop for each item |
| `src/PhotoCatEvo/config.py` | Model settings, extraction limits, default request, and file paths |
| `src/PhotoCatEvo/parameters.json` | Parameter dictionary with aliases, units, plausible ranges, and search terms |
| `src/PhotoCatEvo/parameters.py` | Parameter lookup and the `list`, `show`, and `add` commands |
| `src/PhotoCatEvo/units.py` | Parses units as written in papers (`m2 g−1`, `µmol h-1 g-1`) with Pint, including Pint contexts (eV <-> nm) |
| `src/PhotoCatEvo/conversions.py` | Conversions that need data from the paper: reference electrodes (+ pH), catalyst-mass normalisation, molar mass ([periodictable](https://pypi.org/project/periodictable/)) |
| `src/PhotoCatEvo/standardise.py` | Converts one result to its standard unit using `units.py` and `conversions.py` |
| `src/PhotoCatEvo/collect_papers.py` | Finds open-access photocatalysis papers using the dictionary, saves them to a spreadsheet and downloads the PDFs |
| `src/PhotoCatEvo/sources.py` | Searches OpenAlex and arXiv; finds the PubMed Central open-access copy of a paper |
| `src/PhotoCatEvo/journals.json` | Open-access journals to search (checked against OpenAlex each run). Add your own |
| `src/PhotoCatEvo/pdf_to_text.py` | Converts a PDF and optional supporting information to text |
| `src/PhotoCatEvo/llm.py` | Calls the model endpoint and retries transient failures |
| `src/PhotoCatEvo/tools.py` | Searches paper text, checks evidence and values, converts units, looks up molar masses, and saves results |
| `tests/` | Tests for extraction, evidence checks, sample handling, and unit conversion |
| `examples/text/` | Converted text from two example papers |
| `examples/results/` | Example extraction results in CSV and JSON, with tool-call traces |

`save_result` automatically checks that (1) the quote is really in the paper, (2) the value (and any range or
error) is in its quote and (3) the value is in the expected range, if the parameter dictionary has one for that kind of data (only checked when the paper uses the dictionary's unit).
If a check fails, the result is **rejected** and the agent must fix it. Text answers (e.g. "anatase") are
also accepted, as long as they appear in the quote.

## The parameter dictionary

`parameters.json` lists the experimental quantities the agent knows about. When you run a prompt the agent:

1. gives the planner the list of dictionary names, so "bandgap", "surface area" or "hydrogen evolution rate"
   in a prompt become the dictionary entries `band gap`, `BET surface area`, `H2 evolution rate`;
2. gives each extraction run that entry's notes (what it is, other names, search terms, usual method,
   expected range), so it searches better and doesn't confuse similar quantities;
3. rejects values outside the entry's `plausible_range` (when the paper uses the same unit).

Items that are not in the dictionary still work; the agent just has no notes or range for them, and tells
you so when it runs.

An entry looks like this (every field except the name is optional):

```json
"BET surface area": {
  "aliases": ["surface area", "specific surface area", "SSA", "SBET"],
  "unit": "m2 g-1",
  "plausible_range": [0.1, 3000],
  "search_terms": ["BET", "surface area", "m2 g", "N2 adsorption"],
  "methods": ["N2 adsorption-desorption isotherm, BET method"],
  "description": "Specific surface area from N2 physisorption using the BET method."
}
```

Optional fields for conversions: `standard_unit` (convert to this instead of `unit`), `contexts`
(e.g. `["spectroscopy"]`), `reference` (e.g. `"NHE"`, for potentials) and `species` (e.g. `"H2"`, for rates).

Add entries by editing `src/PhotoCatEvo/parameters.json` or by running the dictionary module from the repository root.

```bash
uv run python -m PhotoCatEvo.parameters list
uv run python -m PhotoCatEvo.parameters show "surface area"
uv run python -m PhotoCatEvo.parameters add "zeta potential" --unit mV --range -100 100 --aliases "surface charge" --search "zeta potential,mV" --methods "Electrophoretic light scattering" --description "Surface charge of particles in suspension."
```

## Unit conversion and standardised units

After all items are extracted, a final step converts every value, range and error to a standard unit for
its kind of measurement. The values exactly as written in the paper stay in `value` / `unit`; the converted
ones go in the `standard_*` columns, and `conversion_note` says what was done.

The standard unit is the dictionary entry's `unit` (or `standard_unit`). Items not in the dictionary go to the
SI base unit (e.g. `ns` -> `s`); dimensionless values and `%` are kept as they are. Conversions are tried in order:

| Kind | Example | How |
|---|---|---|
| Plain unit change | `6.0 mmol g−1 h−1` -> `6000 µmol g-1 h-1`, `38 Å` -> `3.8 nm` | Pint |
| Different kinds of quantity | `455.8 nm` -> `2.72 eV`; `1.94 eV vs. NHE` -> `1.94 V vs. NHE` | Pint contexts, only if the entry allows them (`contexts`) |
| Reference electrodes | `-0.4 V vs. Ag/AgCl` -> `-0.203 V vs. NHE`; `V vs. RHE` -> `V vs. NHE` | `conversions.py`, entry `reference`; RHE needs the **pH** |
| Per gram of catalyst | `300 µmol h-1` -> `6000 µmol g-1 h-1` | `conversions.py`; needs the **catalyst mass** |
| Mass <-> moles of product | `1 mg g-1 h-1` H2 -> `496 µmol g-1 h-1` | molar mass from `periodictable`, entry `species` |

**Values needed from the paper.** If a conversion needs the pH or the catalyst mass, the agent runs once more to
extract it (with the same evidence checks), then converts. These extra results appear as their own rows with
`supporting_for` saying which item they were needed for. If you asked for them in the prompt, that result is reused.
If the value is not in the paper, the item is left unconverted and `conversion_note` says what was missing.

Ranges and errors are converted too. When a conversion is not linear (eV -> nm is 1/x) the range order and
`<`/`>` are flipped, and errors are propagated around the value (|f(x+e) - f(x-e)| / 2).

Pint contexts available: `spectroscopy` (E = hc/λ: eV, nm, cm⁻¹, Hz), `boltzmann` (E = k_B·T), and `electron`
(E = e·V, so "eV vs. NHE" is read as volts). They are only used when the dictionary entry lists them, so a
wavelength is never turned into an energy by accident.

During extraction the agent can also call `convert_unit` (including contexts and reference electrodes, with an
optional pH) and `molar_mass` to compare values. It still saves values as written; standardising is done
afterwards in code.

Limits: the reference offsets assume 25 °C and saturated KCl for Ag/AgCl unless "3 M KCl" is written; one
catalyst mass is used per paper; gas volumes (mL h-1) are not converted to moles.

## Collecting open-access papers

`PhotoCatEvo-collect` (`src/PhotoCatEvo/collect_papers.py`) finds open-access photocatalysis papers, lists them in a spreadsheet and downloads the PDFs,
ready for `PhotoCatEvo`.

```bash
uv run PhotoCatEvo-collect --parameters "band gap" "H2 evolution rate" --from-year 2022 --per-query 10
```

1. **Journals**: the names in `src/PhotoCatEvo/journals.json` (RSC Advances, Nanoscale Advances, Catalysts, Nanomaterials,
   Scientific Reports, Nature Communications, ACS Omega, Advanced Science, ...) are looked up in OpenAlex. Only
   journals OpenAlex lists as open access are searched; the rest are reported and skipped. `--anywhere` searches
   every open-access journal instead.
2. **Queries from the dictionary**: one search per parameter in `src/PhotoCatEvo/parameters.json`, built from its name and
   aliases, e.g. `(photocatalytic OR photocatalysis OR photocatalyst) AND ("band gap" OR "bandgap" OR ...)`.
   Entries with `"search_papers": false` (catalyst mass, pH) are left out. Add a parameter to the dictionary and
   it is searched for next time.
3. **Sources**: [OpenAlex](https://openalex.org) (open-access journal articles) and [arXiv](https://arxiv.org) (preprints).
4. **Spreadsheet**: `results/collected_papers.csv` and `.xlsx`, one row per paper: title, authors, year, journal,
   DOI, licence, which parameters found it, which parameters the abstract mentions, PDF file and where it came from.
   Each run adds to the same spreadsheet, without duplicates.
5. **PDFs** go to `papers/collected/`. The PubMed Central open-access copy is tried first (NIH's official dataset,
   made for automated download), then each open-access copy OpenAlex knows. Some publishers (e.g. Wiley, MDPI)
   refuse automated downloads even of open-access papers; if no other copy exists the row says so and keeps
   `landing_url` to open by hand.

Then: `uv run PhotoCatEvo papers/collected/<file>.pdf "Find the band gap ..."`.

Options: `--per-query`, `--from-year`, `--sources openalex arxiv`, `--max-downloads`, `--no-download`,
`--out`, `--pdf-dir`. Run `uv run PhotoCatEvo-collect --help` for all of them.

**OpenAlex budget**: without a key, everyone on your network shares a small free daily budget (~100 requests;
a run uses one request per parameter, plus one per journal the first time, which is then cached in `.cache/`).
For more, get a free key at https://openalex.org/settings/api and `export OPENALEX_API_KEY=...`.
Only generic search words are sent to these services, never your name or email.

**Licences**: the `license` column shows each paper's licence (e.g. `cc-by`, `cc-by-nc`). Open access means free
to read and download; check the licence before sharing PDFs or reusing figures.

## Setup

Install uv, then run these commands from the repository root. The project requires Python 3.12 or newer. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync --locked
uv run PhotoCatEvo --help
uv run pytest
```

## Run

```bash
# OpenRouter (key from your shell; never put it in a file in this folder)
export LLM_BASE_URL=https://openrouter.ai/api/v1
export OPENROUTER_API_KEY=sk-or-...        # or LLM_API_KEY
export LLM_MODEL=anthropic/claude-opus-5.5 # any slug from https://openrouter.ai/models
export LLM_TEMPERATURE=none                # let reasoning models use their default temperature

# Give it a PDF and a prompt (it converts the PDF itself)
uv run PhotoCatEvo papers/paper_001.pdf "Find the band gap, BET surface area and H2 evolution rate of the best sample"

# With Supporting Information
uv run PhotoCatEvo papers/paper_001.pdf "..." --si papers/paper_001_SI.pdf

# No prompt -> DEFAULT_PROMPT in config.py (the original bandgap task)
uv run PhotoCatEvo papers/paper_001.pdf
```

For a local vLLM instead: `export LLM_BASE_URL=http://localhost:8000/v1` and
`export LLM_MODEL=Qwen/Qwen3.5-9B` (the exact name you used in `vllm serve`).
On Windows PowerShell use `$env:LLM_MODEL="..."` instead of `export`.

If you keep your model settings in the local `.env` file, you can pass it to uv. The bundled text example runs without PDF conversion.

```bash
uv run --env-file .env PhotoCatEvo examples/text/paper_017.txt "Find the band gap and BET surface area."
```

Results:
- `results/paper_001_extraction.json`: results for each data item and sample, with values, units, evidence, and status
- `results/paper_001_extraction.csv`: the same results in tabular form
- `results/paper_001_extraction_trace.json`: the plan and every tool call for every item, useful for debugging and for slides
- `text/paper_001.txt`: the converted paper text

### Output columns

| Column | Meaning |
|---|---|
| `paper`, `doi` | File name and DOI of the paper (DOI found in the text automatically, or given with `--doi`) |
| `sample_name`, `is_main` | **One row per sample.** Every sample made and measured in the paper gets its own row, with its name as written in the paper; `is_main = yes` for the best/main sample |
| `value` | The specific value reported in the paper (preferred) |
| `range` | Fallback when the paper only gives a range or bound, e.g. `2.6-2.8` or `>160`. Can also be filled alongside a value |
| `error` | Uncertainty reported for that data, e.g. `± 0.02`, as written in the paper |
| `standard_value`, `standard_range`, `standard_error`, `standard_unit` | The same data converted to the standard unit for that kind of measurement (see below) |
| `conversion_note` | What conversion was done, or why it could not be (e.g. no unit reported, `a.u.`) |
| `evidence`, `range_evidence`, `error_evidence` | Exact quotes from the paper; each one is checked against the text, and must contain its numbers |

Values from other papers (quotes with `[12]`, `et al.`, `previously reported`, comparison tables) are
rejected, and every sample name must appear in the paper. `warnings` flags values whose quote does not
mention the sample's name, so you can double-check them.

Status values: `verified` (specific value found and checked), `range_only` (no specific value, but a range
or bound was found and checked), `not_reported`, `failed` (no valid answer within 8 steps).

## Cost

The whole paper is sent on every step of every item, so cost grows with (number of items × steps).
With an Anthropic model on OpenRouter the paper text is marked for prompt caching automatically
(`USE_PROMPT_CACHE` in `config.py`), so repeats are much cheaper. Use a cheaper model
(e.g. `anthropic/claude-sonnet-5.5`) while testing.

## vLLM: tool calling must be enabled

```bash
vllm serve Qwen/Qwen3.5-9B --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768
```

Check the Qwen model card / vLLM docs for the recommended `--tool-call-parser` for your exact model version.
A paper without references is ~10k–25k tokens, so `--max-model-len` should be at least 32k.

## Things to try / extend

- Look at the trace: did the planner split the prompt sensibly? Did each run search well and pick the right sample?
- Tune `SYSTEM_PROMPT` in `agent.py` (e.g. which sample counts as "main") and `PLANNER_PROMPT` (how prompts are split).
- Add new quantities to `src/PhotoCatEvo/parameters.json` (or run `uv run python -m PhotoCatEvo.parameters add ...`), or improve the notes and search terms of existing ones.
