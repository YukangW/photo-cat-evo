# Paper data-extraction agent

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

| File | What it does |
|---|---|
| `config.py` | Settings: which LLM, max steps per item, default prompt, where the parameter dictionary is |
| `parameters.json` | **The parameter dictionary**: band gap, BET surface area, ... with aliases, unit, plausible range, search terms, notes. Add your own |
| `parameters.py` | Reads the dictionary, matches prompt items to it, and `list` / `show` / `add` commands |
| `units.py` | Unit conversion with [Pint](https://pint.readthedocs.io): reads units as papers write them (`m2 g−1`, `µmol h-1 g-1`) and standardises results |
| `pdf_to_text.py` | PDF (+ SI) → `text/<name>.txt` (drops references and the vertical publisher watermark) |
| `llm.py` | The only file that calls the LLM (vLLM or OpenRouter), with retries |
| `tools.py` | Planning tool `set_targets`, and extraction tools `search_text`, `check_evidence`, `check_plausible`, `convert_unit`, `save_result` |
| `agent.py` | Plans the items, then runs the tool loop once per item until `save_result` is accepted (max 8 steps each) |

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

Add entries by editing `parameters.json`, or from the command line:

```bash
python parameters.py list
python parameters.py show "surface area"
python parameters.py add "zeta potential" --unit mV --range -100 100 --aliases "surface charge" --search "zeta potential,mV" --methods "Electrophoretic light scattering" --description "Surface charge of particles in suspension."
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

```bash
pip install -r requirements.txt
```

## Run

```bash
# OpenRouter (key from your shell; never put it in a file in this folder)
export LLM_BASE_URL=https://openrouter.ai/api/v1
export OPENROUTER_API_KEY=sk-or-...        # or LLM_API_KEY
export LLM_MODEL=anthropic/claude-opus-5.5 # any slug from https://openrouter.ai/models
export LLM_TEMPERATURE=none                # let reasoning models use their default temperature

# Give it a PDF and a prompt (it converts the PDF itself)
python agent.py papers/paper_001.pdf "Find the band gap, BET surface area and H2 evolution rate of the best sample"

# With Supporting Information
python agent.py papers/paper_001.pdf "..." --si papers/paper_001_SI.pdf

# No prompt -> DEFAULT_PROMPT in config.py (the original bandgap task)
python agent.py papers/paper_001.pdf
```

For a local vLLM instead: `export LLM_BASE_URL=http://localhost:8000/v1` and
`export LLM_MODEL=Qwen/Qwen3.5-9B` (the exact name you used in `vllm serve`).
On Windows PowerShell use `$env:LLM_MODEL="..."` instead of `export`.

Results:
- `results/paper_001_extraction.json`: one entry per data item (value, range, error, unit, sample, quotes, status)
- `results/paper_001_extraction.csv`: the same as a table, one row per data item (opens in Excel)
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
- Add new quantities to `parameters.json` (or `python parameters.py add ...`), or improve the notes and search terms of existing ones.
