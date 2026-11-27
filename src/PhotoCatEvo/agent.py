"""
Data-extraction agent: reads one paper and extracts the data asked for in a prompt.

How it works
  1. Plan:    the LLM splits the prompt into separate data items ("band gap", "BET surface area", ...).
  2. Extract: the agent runs once PER ITEM, in order. Each run searches the paper with tools and must
              pass the evidence checks in save_result before it moves on to the next item.
  3. Standardise: after extraction, every value/range/error is converted to the standard unit for its
              kind of measurement (from parameters.json; SI base unit if it is not in there), using Pint,
              Pint contexts (eV <-> nm) and conversions.py (reference electrodes, catalyst mass, molar mass).
              If a conversion needs another value from the paper (pH, catalyst mass) the agent runs again
              to extract it first. The values as written in the paper are kept; converted ones go in extra columns.

Usage:
    uv run PhotoCatEvo papers/paper_001.pdf "Find the band gap and the BET surface area of the best sample"
    uv run PhotoCatEvo papers/paper_001.pdf "..." --si papers/paper_001_SI.pdf
    uv run PhotoCatEvo text/paper_001.txt "..."          # already-converted text also works
    uv run PhotoCatEvo papers/paper_001.pdf              # no prompt -> DEFAULT_PROMPT in config.py (bandgap)
    uv run PhotoCatEvo papers/collected "..."            # recursively extract PDFs and TXTs, then summarize
    uv run PhotoCatEvo papers/collected "..." --output-dir custom_results
    uv run PhotoCatEvo papers/collected "..." --workers 5
Output:
    results/paper_001_extraction.json        results for each data item and sample
    results/paper_001_extraction.csv         the same as a table: item, value, range, error, unit,
                                             standard_value/_range/_error/_unit, sample, ...
    results/paper_001_extraction_trace.json  the plan, tool calls and extraction metrics
    results/cost_metrics.csv                time and tokens by paper, planning stage and parameter (directory mode)
"""
import argparse
import csv
import json
import os
import re
import sys
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter

from .config import DEFAULT_PROMPT, MAX_STEPS, MAX_TARGETS, RESULTS_DIR, TEXT_DIR
from .llm import TokenUsage, chat, model_settings
from .organize import organize
from .parameters import describe_for_extractor, describe_for_planner, find_parameter, load_parameters
from .standardise import standardise
from .tools import PLAN_TOOLS, TOOLS, PaperTools, PlanTools
from .units import UnitError, convert

_EXTRACTED_STATUSES = {"verified", "range_only"}


@dataclass(frozen=True)
class PaperProgress:
    stage: str
    completed: int | None = None
    total: int | None = None
    rows: list[dict] | None = None
    result_path: Path | None = None
    error: str | None = None
    current_parameter: str | None = None


PLANNER_PROMPT = """You turn a user's data-extraction request into a list of separate data items.

Rules:
1. One item per distinct quantity the user asks for, in the order they ask for them.
   "the band gap and BET surface area" -> two items.
2. Do NOT split one quantity by sample; the extractor reports other samples itself.
   Only make separate items for the same quantity if the user explicitly asks for it under
   different conditions (e.g. "H2 evolution rate under UV and under visible light" -> two items).
3. Put anything the user specified about an item (which sample, conditions, method) in "details".
4. Do not add items the user did not ask for.
5. If an item is in the parameter dictionary below (by name or by one of its other names), use the
   dictionary name exactly as "name" and its unit as "unit". Items not in the dictionary are still fine:
   give them a short clear name.
Call set_targets with the list."""

SYSTEM_PROMPT = """You are an expert in materials science and photocatalysis who extracts data from papers.
You extract ONE data item at a time; the item is given in the user's message.

Rules:
1. Report the value for EVERY sample that was made and measured in THIS paper, each with its name
   exactly as written in the paper (e.g. "CN-550", "bulk g-C3N4", "P25"). Pristine or reference
   samples count if the authors measured them themselves.
   - Put the MAIN sample (the optimised/best one the paper focuses on, unless the item's details name
     a sample) in value / sample_name, and every other sample in other_samples.
   - Use the same sample names as earlier items of this paper (they are listed in the user message).
   - Material properties (band gap, surface area, XRD, ...) are often measured BEFORE the cocatalyst is
     added: e.g. the band gap is given for CNU20 while the best H2 sample is Pt-CNU20. Then report the
     values of the samples the paper actually measured (CNU20 as main sample); do not search for a value
     of the loaded sample that the paper does not give.
   - NEVER report values from other papers: statements in the introduction, values of other works
     in comparison tables, or anything with a citation ([12], "et al.", "previously reported").
     Quotes with citations are rejected automatically.
2. Every value needs an EXACT quote copied from the paper that contains it. One quote can cover
   several samples (a table row, or "the Eg of A and B is 2.70 and 2.75 eV, respectively").
3. Report the value exactly as written in the quote. Do not guess, convert units or calculate.
4. Specific value first, range as a fallback:
   - If the paper gives a specific value, put it in "value".
   - If it only gives a range or a bound (e.g. "2.6-2.8 eV", ">160 m2 g-1", "about 3 eV or less"),
     set value = null and copy the range into "range" as written (e.g. "2.6-2.8", ">160").
   - If it gives both a value and a range, fill both.
5. Errors: if an uncertainty is reported for this data (e.g. "2.72 ± 0.02 eV", error bars given as
   numbers, standard deviation), copy it into "error" as written (e.g. "± 0.02"). Otherwise leave it ''.
6. If neither a value nor a range is reported for any sample, save value = null and range = ''.
   Values only shown in a figure (no number in the text) count as not reported.
7. Use the tools: search_text to find candidate sentences (try synonyms, units and method names),
   check_evidence / check_plausible if unsure, and finish by calling save_result.
   If save_result is rejected, fix the problem and try again.
8. The user's message may include notes on the item from the parameter dictionary (what it is, other
   names, search terms, expected range). Use them to search well and to avoid confusing similar quantities.
"""


# ============================================================ helpers
def _assistant_message(msg) -> dict:
    """Copy the model's reply into the conversation, keeping any reasoning the provider sent.
    (Reasoning models on OpenRouter need their reasoning passed back alongside tool calls.)"""
    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
    for key in ("reasoning_details", "reasoning"):
        value = (msg.model_extra or {}).get(key)
        if value:
            out[key] = value
    return out


def _paper_message(paper_text: str) -> dict:
    """The paper itself. Kept identical across every request so the provider can cache it."""
    text = f"Paper text:\n\n{paper_text}"
    if model_settings.get().use_prompt_cache:
        return {"role": "user",
                "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}
    return {"role": "user", "content": text}


def _describe(target: dict) -> str:
    s = target["name"]
    if target.get("unit"):
        s += f" ({target['unit']})"
    if target.get("details"):
        s += f" - {target['details']}"
    return s


def _summary(target: dict, result: dict) -> str:
    if result.get("status") not in _EXTRACTED_STATUSES:
        return f"- {target['name']}: {result.get('status')}"
    shown = result["value"] if result.get("value") is not None else f"range {result['range']}"
    if result.get("error"):
        shown = f"{shown} (error {result['error']})"
    sample = f" for sample {result['sample_name']}" if result.get("sample_name") else ""
    line = f"- {target['name']}: {shown} {result.get('unit', '')}{sample}".rstrip()
    others = result.get("other_samples") or []
    if others:
        line += "; other samples: " + ", ".join(
            f"{o['sample_name']} {o['value'] if o.get('value') is not None else o.get('range', '')}"
            for o in others)
    return line


def _sample_names(result: dict) -> list[str]:
    names = [result.get("sample_name")] + [o.get("sample_name") for o in result.get("other_samples") or []]
    return [n for n in names if n]


def find_doi(text: str) -> str:
    """The paper's DOI, from the first pages (header / footer: 'https://doi.org/10....', 'DOI: 10....')."""
    head = text[:15000]
    candidates = re.findall(r"(?:doi\.org/|doi:?\s*)(10\.\d{4,9}/\S+)", head, re.IGNORECASE) \
        or re.findall(r"\b(10\.\d{4,9}/[^\s\"<>]+)", head)
    if not candidates:
        return ""
    return candidates[0].rstrip(".,;:)]}'\"")


def _rows(target: dict, result: dict, paper: str, doi: str) -> list[dict]:
    """One row per sample: the main sample first (is_main = yes), then the other samples."""
    base = {"paper": paper, "doi": doi, "item": target["name"], "parameter": target["parameter"],
            "requested_unit": target.get("unit", ""), "details": target.get("details", ""),
            "steps": result.get("steps"), "warnings": "; ".join(result.get("warnings") or [])}
    main = {k: v for k, v in result.items() if k not in ("other_samples", "warnings", "steps")}
    rows = [{**base, **main, "is_main": "yes" if result.get("status") in _EXTRACTED_STATUSES else ""}]
    for o in result.get("other_samples") or []:
        rows.append({**base, **o, "is_main": "no"})
    return rows


# Columns of the results table (results/<paper>_extraction.csv)
CSV_COLUMNS = ["paper", "doi", "item", "parameter", "sample_name", "is_main", "value", "range", "error", "unit",
               "standard_value", "standard_range", "standard_error", "standard_unit", "conversion_note",
               "status", "method", "evidence", "range_evidence", "error_evidence", "note", "warnings",
               "supporting_for"]


def _write_csv(path: str, results: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig so Excel shows µ, ±, −
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            # one line per cell: drop the PDF's line breaks and soft hyphens from quotes
            writer.writerow({col: " ".join(str(r.get(col) if r.get(col) is not None else "")
                                           .replace("­\n", "").split())
                             for col in CSV_COLUMNS})


def _print_table(results: list[dict]) -> None:
    rows = [["item", "sample_name", "is_main", "value", "range", "error", "unit", "standard_value",
             "standard_unit", "status"]]
    rows += [[str(r.get(c) if r.get(c) is not None else "") for c in rows[0]] for r in results]
    widths = [min(30, max(len(row[i]) for row in rows)) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell[:30].ljust(w) for cell, w in zip(row, widths)))


def _format_token_usage(usage: dict) -> str:
    reported = usage["requests_with_usage"]
    missing = usage["requests_without_usage"]
    if not reported and missing:
        return f"token usage unavailable (0/{missing} requests reported usage)"
    counts = (f"{usage['prompt_tokens']:,} input, {usage['completion_tokens']:,} output, "
              f"{usage['total_tokens']:,} total tokens")
    if missing:
        counts += f" (partial, {reported}/{reported + missing} requests reported usage)"
    return counts


def _run_tool_calls(msg, tools, messages: list, trace: list, step: int):
    """Execute every tool call in the model's reply and add the results to the conversation."""
    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
            result = tools.run(name, args)
        except json.JSONDecodeError:
            args, result = tc.function.arguments, {"error": "Arguments were not valid JSON."}

        print(f"  [step {step}] {name}({json.dumps(args, ensure_ascii=False)[:120]})")
        print(f"             -> {json.dumps(result, ensure_ascii=False)[:200]}")
        trace.append({"step": step, "tool": name, "args": args, "result": result})
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(result, ensure_ascii=False)})


# ============================================================ 1. plan
def plan_targets(prompt: str, params: dict | None = None, *, token_usage: TokenUsage | None = None) -> tuple[list, list]:
    """Split the prompt into separate data items. Falls back to the whole prompt as one item."""
    params = load_parameters() if params is None else params
    plan = PlanTools(MAX_TARGETS)
    system = PLANNER_PROMPT
    if params:
        system += "\n\nParameter dictionary:\n" + describe_for_planner(params)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Request: {prompt}"}]
    trace = []

    for step in range(1, 4):
        msg = chat(messages, tools=PLAN_TOOLS, token_usage=token_usage)
        messages.append(_assistant_message(msg))
        if not msg.tool_calls:
            trace.append({"step": step, "text": msg.content})
            messages.append({"role": "user", "content": "Please call set_targets with the list."})
            continue
        _run_tool_calls(msg, plan, messages, trace, step)
        if plan.targets:
            return plan.targets, trace

    print("  [plan] the model did not return a list; treating the whole prompt as one item.")
    return [{"name": prompt, "unit": "", "details": ""}], trace


# ============================================================ 2. extract one item
def extract_one(paper_text: str, target: dict, prompt: str, done_so_far: list[str],
                params: dict | None = None, known_samples: list[str] | None = None, *,
                token_usage: TokenUsage | None = None) -> tuple[dict, list]:
    """Run the tool-using agent loop for ONE data item. Returns (result, trace)."""
    params = load_parameters() if params is None else params
    tools = PaperTools(paper_text, target, params)
    task = [f"Original request: {prompt}",
            f"\nExtract this data item now: {_describe(target)}"]
    if target.get("parameter") in params:
        task.append("\n" + describe_for_extractor(target["parameter"], params[target["parameter"]]))
    else:
        task.append("\nThis item is not in the parameter dictionary. Still extract it: work out from your own "
                    "knowledge what the quantity is, its usual units, other names it goes by and how it is "
                    "measured, and search the paper for all of those. The same evidence rules apply. "
                    "There is no expected range, so check the value makes physical sense yourself.")
    if done_so_far:
        task.append("\nAlready extracted from this paper (for context, e.g. which sample is the main one):\n"
                    + "\n".join(done_so_far))
    if known_samples:
        task.append("\nSample names already used for this paper (reuse exactly these names for the same "
                    "samples): " + ", ".join(known_samples))
    task.append("\nReport EVERY sample of this paper that has a value for this item (main sample + "
                "other_samples). Finish by calling save_result for this item only.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                _paper_message(paper_text),
                {"role": "user", "content": "\n".join(task)}]
    trace = []

    for step in range(1, MAX_STEPS + 1):
        try:
            msg = chat(messages, tools=TOOLS, token_usage=token_usage)
        except RuntimeError as e:          # LLM down after retries: mark this item failed, keep the others
            print(f"  [step {step}] LLM failed: {e}. Skipping this item.")
            trace.append({"step": step, "error": str(e)})
            break
        messages.append(_assistant_message(msg))

        if not msg.tool_calls:             # answered in text without a tool -> nudge it
            print(f"  [step {step}] model replied without a tool call: {(msg.content or '')[:150]!r}")
            trace.append({"step": step, "text": msg.content})
            messages.append({"role": "user",
                             "content": "Please call save_result with your final answer (or value=null)."})
            continue

        _run_tool_calls(msg, tools, messages, trace, step)
        if tools.result is not None:       # save_result accepted -> this item is done
            break
        left = MAX_STEPS - step
        if left in (1, 2):                 # running out of steps: make it save what it has
            messages.append({"role": "user", "content": (
                f"Only {left} step{'s' if left > 1 else ''} left for this item. If you already found quotes with "
                f"values, call save_result NOW with them (main sample + other_samples). Do not keep searching. "
                f"If nothing was found, save value=null.")})

    result = tools.result or {"value": None, "status": "failed",
                              "note": f"No valid result after {MAX_STEPS} steps."}
    result["steps"] = step
    return result, trace


def _extract_with_metrics(paper_text: str, target: dict, prompt: str, done_so_far: list[str],
                          params: dict, paper_usage: TokenUsage,
                          known_samples: list[str] | None = None) -> tuple[dict, list, dict]:
    item_usage = TokenUsage()
    item_started = perf_counter()
    result, trace = extract_one(paper_text, target, prompt, done_so_far, params, known_samples,
                                token_usage=item_usage)
    metrics = {"elapsed_seconds": perf_counter() - item_started, "status": result["status"],
               "token_usage": item_usage.as_dict()}
    paper_usage.add(item_usage)
    return result, trace, metrics


# ============================================================ whole paper
def match_parameters(targets: list[dict], params: dict) -> list[dict]:
    """Attach the matching dictionary entry's name to each item ('' if it is not in the dictionary)."""
    matched = []
    for t in targets:
        match = find_parameter(t["name"], params)
        matched.append({**t, "parameter": match[0] if match else ""})
    return matched


# Values some conversions need, and how to ask for them: need -> (dictionary entry, details for the extractor)
SUPPORTING = {
    "pH": ("electrolyte pH", "pH of the electrolyte used in the electrochemical measurement that gave the {items}. "
                             "Needed to convert potentials between the RHE and NHE scales."),
    "catalyst mass": ("catalyst mass", "mass of photocatalyst used in the activity test that gave the {items}. "
                                       "Needed to normalise the rate per gram of catalyst."),
}


def _supporting_value(need: str, result: dict):
    """The usable value of a supporting result: pH as a number, catalyst mass as (value, unit); else None."""
    value = result.get("value")
    if result.get("status") != "verified" or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if need == "pH":
        return float(value) if 0 <= value <= 14 else None
    try:
        convert(1.0, result.get("unit") or "", "g")         # must be a mass
    except UnitError:
        return None
    return float(value), result["unit"]


def standardise_results(paper_text: str, prompt: str, results: list[dict], params: dict,
                        done_so_far: list[str], *, paper: str, doi: str,
                        paper_usage: TokenUsage) -> tuple[list[dict], list[dict]]:
    """Step 3, after extraction: add the standard_* columns to every result (the originals are kept).

    If a conversion needs a value from the paper (pH for RHE potentials, catalyst mass for per-gram rates),
    the agent runs once more to extract it, then converts with it. Returns (results, traces of extra runs).
    """
    first = [standardise(r, params.get(r.get("parameter"))) for r in results]
    needs: dict[str, list[str]] = {}
    for r, std in zip(results, first):
        for need in std["needs"]:
            needs.setdefault(need, []).append(r["item"])

    supporting, extra, traces = {}, [], []
    for need, items in needs.items():
        name, details = SUPPORTING[need]
        found = next((r for r in results if r.get("parameter") == name), None)   # already asked for?
        if found is None:
            target = {"name": name, "unit": params.get(name, {}).get("unit", ""),
                      "details": details.format(items=", ".join(items)), "parameter": name if name in params else ""}
            print(f"\n[supporting] {name}: needed to convert {', '.join(items)}")
            result, trace, metrics = _extract_with_metrics(paper_text, target, prompt, done_so_far, params,
                                                           paper_usage)
            print(f"  => {_summary(target, result)[2:]}")
            found = {"paper": paper, "doi": doi, "item": name, "parameter": target["parameter"],
                     "requested_unit": target["unit"], "details": target["details"], **result,
                     "supporting_for": ", ".join(items)}
            extra.append(found)
            traces.append({"item": f"{name} (supporting)", "trace": trace, "metrics": metrics})
        value = _supporting_value(need, found)
        if value is not None:
            supporting[need] = value

    print("\nStandardised units:")
    out = []
    for r in results + extra:
        std = standardise(r, params.get(r.get("parameter")), supporting)
        std.pop("needs")
        print(f"  {r['item']}: {std['conversion_note'] or r.get('status')}")
        out.append({**r, **std})
    return out, traces


def run_agent(paper_text: str, paper_name: str, prompt: str = DEFAULT_PROMPT,
              params: dict | None = None, doi: str | None = None, *, output_prefix: Path | None = None,
              pdf_conversion_elapsed: float = 0.0,
              on_progress: Callable[[PaperProgress], None] | None = None) -> dict:
    params = load_parameters() if params is None else params
    doi = doi or find_doi(paper_text)
    print(f"DOI: {doi or 'not found in the text (give it with --doi)'}")
    print(f"Request: {prompt}\nParameter dictionary: {len(params)} entries\n\nPlanning...")
    if on_progress:
        on_progress(PaperProgress("planning"))
    paper_usage = TokenUsage()
    planning_started = perf_counter()
    targets, plan_trace = plan_targets(prompt, params, token_usage=paper_usage)
    targets = match_parameters(targets, params)
    planning_metrics = {"elapsed_seconds": perf_counter() - planning_started, "token_usage": paper_usage.as_dict()}
    print(f"  {len(targets)} data item(s): " + "; ".join(_describe(t) for t in targets))
    for t in targets:
        if t["parameter"]:
            print(f"    '{t['name']}' -> dictionary entry '{t['parameter']}'")
        else:
            print(f"    '{t['name']}' is not in the dictionary: extracting it anyway, without notes or a "
                    f"range check (add it with: uv run python -m PhotoCatEvo.parameters add ...)")

    results, item_traces, done_so_far, known_samples = [], [], [], []
    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {_describe(target)}")
        if on_progress:
            on_progress(PaperProgress("extracting", completed=i - 1, total=len(targets),
                                      current_parameter=target["name"], rows=results.copy()))
        result, trace, item_metrics = _extract_with_metrics(paper_text, target, prompt, done_so_far, params,
                                                            paper_usage, known_samples)
        print(f"  => {_summary(target, result)[2:]}")
        results += _rows(target, result, paper_name, doi)
        item_traces.append({"item": target["name"], "trace": trace, "metrics": item_metrics})
        done_so_far.append(_summary(target, result))
        known_samples += [n for n in _sample_names(result) if n not in known_samples]
    planning_and_extraction_elapsed = perf_counter() - planning_started

    extracted_samples = {r["sample_name"] for r in results
                         if r.get("sample_name") and r["status"] in _EXTRACTED_STATUSES}
    print("\nStandardising units...")
    if on_progress:
        on_progress(PaperProgress("standardizing", completed=len(targets), total=len(targets), rows=results.copy()))
    results, extra_traces = standardise_results(paper_text, prompt, results, params, done_so_far,
                                               paper=paper_name, doi=doi, paper_usage=paper_usage)
    supporting_elapsed = sum(item["metrics"]["elapsed_seconds"] for item in extra_traces)
    metrics = {"elapsed_seconds": pdf_conversion_elapsed + planning_and_extraction_elapsed + supporting_elapsed,
               "parameters_attempted": len(item_traces),
               "parameters_extracted": sum(item["metrics"]["status"] in _EXTRACTED_STATUSES
                                           for item in item_traces),
               "sample_count": len(extracted_samples), "token_usage": paper_usage.as_dict()}

    final = {"paper": paper_name, "doi": doi, "model": model_settings.get().model, "prompt": prompt,
             "samples": known_samples, "results": results}

    output_prefix = output_prefix or Path(RESULTS_DIR) / paper_name
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{output_prefix}_extraction.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    with open(f"{output_prefix}_extraction_trace.json", "w", encoding="utf-8") as f:
        json.dump({"plan": {"targets": targets, "trace": plan_trace, "metrics": planning_metrics},
                   "items": item_traces + extra_traces,
                   "metrics": metrics},
                  f, indent=2, ensure_ascii=False)
    _write_csv(f"{output_prefix}_extraction.csv", results)

    print("\nEXTRACTION METRICS")
    print(f"  Whole paper: {metrics['parameters_attempted']} parameters attempted, "
          f"{metrics['parameters_extracted']} extracted, {metrics['sample_count']} samples, "
          f"{metrics['elapsed_seconds']:.2f} s, {_format_token_usage(metrics['token_usage'])}")
    print(f"  Planning: {planning_metrics['elapsed_seconds']:.2f} s, "
          f"{_format_token_usage(planning_metrics['token_usage'])}")
    labels = [_describe(target) for target in targets] + [item["item"] for item in extra_traces]
    for label, item in zip(labels, item_traces + extra_traces):
        item_metrics = item["metrics"]
        print(f"  {label} [{item_metrics['status']}]: {item_metrics['elapsed_seconds']:.2f} s, "
              f"{_format_token_usage(item_metrics['token_usage'])}")
    return final


def load_paper(path: str, si_path: str | None = None, *, save_text: bool = True) -> tuple[str, str]:
    """Return (text, name), optionally saving converted PDF text in text/."""
    text, name, _ = load_paper_with_timing(path, si_path, save_text=save_text)
    return text, name


def load_paper_with_timing(path: str, si_path: str | None = None, *, save_text: bool = True) -> tuple[str, str, float]:
    name = os.path.splitext(os.path.basename(path))[0]
    pdf_conversion_elapsed = 0.0
    if path.lower().endswith(".pdf"):
        from .pdf_to_text import convert          # needs pymupdf, only imported for PDFs
        conversion_started = perf_counter()
        text, n_tables = convert(path, si_path)
        pdf_conversion_elapsed = perf_counter() - conversion_started
        print(f"Converted PDF: {n_tables} tables found")
        if save_text:
            os.makedirs(TEXT_DIR, exist_ok=True)
            with open(os.path.join(TEXT_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    return text, name, pdf_conversion_elapsed


def _cost_rows(paper_name: str, trace: dict) -> list[tuple]:
    entries = [("paper", "", trace["metrics"]), ("planning", "", trace["plan"]["metrics"])]
    entries.extend(("parameter", item["item"], item["metrics"]) for item in trace["items"])
    rows = []
    for scope, parameter, metrics in entries:
        usage = metrics["token_usage"]
        rows.append((paper_name, scope, parameter, metrics["elapsed_seconds"],
                     usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]))
    return rows


def run_directory(folder: Path, prompt: str, results_dir: Path, workers: int = 1, *,
                  on_paper_progress: Callable[[str, PaperProgress], None] | None = None,
                  on_summarizing: Callable[[], None] | None = None) -> tuple[int, int]:
    """Process a directory and return the numbers of successful and failed papers."""
    if results_dir.resolve() == folder.resolve():
        raise ValueError("The output directory must differ from the input directory, since paper filenames become directories.")

    papers = sorted(
        (path for path in folder.rglob("*") if path.suffix.lower() in {".pdf", ".txt"} and path.is_file()),
        key=Path.as_posix,
    )
    if not papers:
        raise ValueError(f"No PDF or TXT files found in '{folder}'.")

    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {model_settings.get().model}\nResults directory: {results_dir}")
    cost_rows = {}
    index_width = max(4, len(str(len(papers))))
    inputs = iter(enumerate(papers, start=1))
    with TemporaryDirectory(dir=results_dir) as temp_dir, ThreadPoolExecutor(max_workers=workers) as pool:
        batch_dir = Path(temp_dir)
        pending = {}
        while True:
            for index, path in inputs:
                relative_path = path.relative_to(folder)
                paper_name = relative_path.as_posix()
                print(f"\n[{index}/{len(papers)}] {paper_name}")
                if on_paper_progress:
                    on_paper_progress(paper_name, PaperProgress("reading"))
                try:
                    text, _, pdf_conversion_elapsed = load_paper_with_timing(str(path), save_text=False)
                except Exception as exc:
                    print(f"Failed to process {paper_name}: {exc}", file=sys.stderr)
                    if on_paper_progress:
                        on_paper_progress(paper_name, PaperProgress("failed", error=str(exc)))
                    continue

                output_prefix = batch_dir / "work" / str(index)
                report = partial(on_paper_progress, paper_name) if on_paper_progress else None
                future = pool.submit(copy_context().run, run_agent, text, paper_name, prompt, output_prefix=output_prefix,
                                     pdf_conversion_elapsed=pdf_conversion_elapsed, on_progress=report)
                pending[future] = (index, relative_path, output_prefix)
                if len(pending) == workers:
                    break

            if not pending:
                break

            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index, relative_path, output_prefix = pending.pop(future)
                paper_name = relative_path.as_posix()
                paper_dir = results_dir / relative_path
                result_path = paper_dir / f"{relative_path.stem}_extraction.json"
                batch_csv = batch_dir / f"{index:0{index_width}d}_extraction.csv"
                try:
                    future.result()
                    with open(f"{output_prefix}_extraction_trace.json", encoding="utf-8") as trace_file:
                        paper_cost_rows = _cost_rows(paper_name, json.load(trace_file))
                    paper_dir.mkdir(parents=True, exist_ok=True)
                    copyfile(f"{output_prefix}_extraction.csv", batch_csv)
                    for suffix in ("_extraction.json", "_extraction_trace.json", "_extraction.csv"):
                        Path(f"{output_prefix}{suffix}").replace(paper_dir / f"{relative_path.stem}{suffix}")
                    cost_rows[index] = paper_cost_rows
                except Exception as exc:
                    batch_csv.unlink(missing_ok=True)
                    print(f"Failed to process {paper_name}: {exc}", file=sys.stderr)
                    if on_paper_progress:
                        on_paper_progress(paper_name, PaperProgress("failed", error=str(exc)))
                else:
                    if on_paper_progress:
                        on_paper_progress(paper_name, PaperProgress("succeeded", result_path=result_path))

        failed = len(papers) - len(cost_rows)
        if failed == len(papers):
            return 0, failed

        if on_summarizing:
            on_summarizing()
        summary = Path(organize(str(batch_dir), recursive=False))
        cost_csv = batch_dir / "cost_metrics.csv"
        with cost_csv.open("w", encoding="utf-8-sig", newline="") as cost_file:
            writer = csv.writer(cost_file)
            writer.writerow(("paper", "scope", "parameter", "elapsed_seconds",
                             "prompt_tokens", "completion_tokens", "total_tokens"))
            for index in sorted(cost_rows):
                writer.writerows(cost_rows[index])
        summary = summary.replace(results_dir / summary.name)
        cost_report = cost_csv.replace(results_dir / cost_csv.name)
    print(f"Batch summary: {summary}")
    print(f"Cost metrics: {cost_report}")
    return len(cost_rows), failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract data from one paper or a directory of papers.",
                                     epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", help="A PDF, a TXT, or a directory to search recursively for both")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="What data to extract")
    parser.add_argument("--si", help="Supporting Information PDF (optional)")
    parser.add_argument("--doi", help="The paper's DOI, if it is not found in the text automatically")
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR),
                        help="Directory for extraction results and the batch summary (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Maximum papers to extract concurrently in directory mode (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    paper_path = Path(args.paper)
    if paper_path.is_dir():
        if args.si is not None or args.doi is not None:
            parser.error("--si and --doi apply to a single paper only")
        try:
            succeeded, failed = run_directory(paper_path, args.prompt, args.output_dir, args.workers)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if failed and not succeeded:
            raise SystemExit(f"All {failed} papers failed.")
        if failed:
            raise SystemExit(f"{failed} of {succeeded + failed} papers failed.")
        return

    text, name, pdf_conversion_elapsed = load_paper_with_timing(args.paper, args.si)
    print(f"Model: {model_settings.get().model}\nPaper: {name} ({len(text):,} chars)\n")
    output_prefix = args.output_dir / name
    final = run_agent(text, name, args.prompt, doi=args.doi, output_prefix=output_prefix,
                      pdf_conversion_elapsed=pdf_conversion_elapsed)
    print("\nFINAL RESULTS:")
    _print_table(final["results"])
    print(f"\nFull results (with evidence quotes): {output_prefix}_extraction.json and .csv")


if __name__ == "__main__":
    main()
