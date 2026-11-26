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
| `src/PhotoCatEvo/units.py` | Parses units as written in papers and standardises results with Pint |
| `src/PhotoCatEvo/pdf_to_text.py` | Converts a PDF and optional supporting information to text |
| `src/PhotoCatEvo/llm.py` | Calls the model endpoint and retries transient failures |
| `src/PhotoCatEvo/tools.py` | Searches paper text, checks evidence and values, and saves results |
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

Add entries by editing `src/PhotoCatEvo/parameters.json` or by running the dictionary module from the repository root.

```bash
uv run python -m PhotoCatEvo.parameters list
uv run python -m PhotoCatEvo.parameters show "surface area"
uv run python -m PhotoCatEvo.parameters add "zeta potential" --unit mV --range -100 100 --aliases "surface charge" --search "zeta potential,mV" --methods "Electrophoretic light scattering" --description "Surface charge of particles in suspension."
```

## Unit conversion and standardised units

After all items are extracted, a final step converts every value, range and error with Pint to a standard unit
for its kind of measurement, e.g. `6.0 mmol g−1 h−1` -> `6000 µmol g-1 h-1`, `38 Å` -> `3.8 nm`.
The values exactly as written in the paper stay in `value` / `unit`; the converted ones go in the
`standard_*` columns.

- The standard unit is the dictionary entry's `unit` (or `standard_unit`, if you set one).
- Items not in the dictionary are converted to the SI base unit (e.g. `ns` -> `s`); dimensionless values and
  `%` are kept as they are.
- The range check (`check_plausible`) also uses Pint, so a value in mmol is compared correctly with a range in µmol.
- During extraction the agent also has a `convert_unit` tool (e.g. `5e-9 m` -> `5 nm`) for comparing values
  given in different units. It still saves values as written; standardising is done afterwards in code.

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
