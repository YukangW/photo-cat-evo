"""
Data-extraction agent: reads one paper and extracts the data asked for in a prompt.

How it works
  1. Plan:    the LLM splits the prompt into separate data items ("band gap", "BET surface area", ...).
  2. Extract: the agent runs once PER ITEM, in order. Each run searches the paper with tools and must
              pass the evidence checks in save_result before it moves on to the next item.
  3. Standardise: after extraction, every value/range/error is converted with Pint to the standard unit
              for its kind of measurement (from parameters.json; SI base unit if it is not in there).
              The values as written in the paper are kept; the converted ones go in extra columns.

Usage:
    python agent.py papers/paper_001.pdf "Find the band gap and the BET surface area of the best sample"
    python agent.py papers/paper_001.pdf "..." --si papers/paper_001_SI.pdf
    python agent.py text/paper_001.txt "..."          # already-converted text also works
    python agent.py papers/paper_001.pdf              # no prompt -> DEFAULT_PROMPT in config.py (bandgap)
Output:
    results/paper_001_extraction.json        one result per data item
    results/paper_001_extraction.csv         the same as a table: item, value, range, error, unit,
                                             standard_value/_range/_error/_unit, sample, ...
    results/paper_001_extraction_trace.json  the plan + every tool call for every item (debugging/slides)
"""
import argparse
import csv
import json
import os

from config import (DEFAULT_PROMPT, MAX_STEPS, MAX_TARGETS, MODEL, RESULTS_DIR, TEXT_DIR,
                    USE_PROMPT_CACHE)
from llm import chat
from parameters import (describe_for_extractor, describe_for_planner, find_parameter, load_parameters,
                        standard_unit_for)
from tools import PLAN_TOOLS, TOOLS, PaperTools, PlanTools
from units import standardise

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
1. Report the value for the MAIN sample: the optimised/best sample the paper focuses on
   (e.g. the one with the best photocatalytic performance), unless the item's details name a sample.
   If several samples have values, put the others in all_values.
2. The evidence must be an EXACT quote copied from the paper that contains the value.
3. Report the value exactly as written in the quote. Do not guess, convert units or calculate.
4. Specific value first, range as a fallback:
   - If the paper gives a specific value, put it in "value".
   - If it only gives a range or a bound (e.g. "2.6-2.8 eV", ">160 m2 g-1", "about 3 eV or less"),
     set value = null and copy the range into "range" as written (e.g. "2.6-2.8", ">160").
   - If it gives both a value and a range, fill both.
5. Errors: if an uncertainty is reported for this data (e.g. "2.72 ± 0.02 eV", error bars given as
   numbers, standard deviation), copy it into "error" as written (e.g. "± 0.02"). Otherwise leave it ''.
6. If neither a value nor a range is reported in the text, save value = null and range = ''.
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
    if USE_PROMPT_CACHE:
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
    if result.get("status") not in ("verified", "range_only"):
        return f"- {target['name']}: {result.get('status')}"
    shown = result["value"] if result.get("value") is not None else f"range {result['range']}"
    if result.get("error"):
        shown = f"{shown} (error {result['error']})"
    sample = f" for sample {result['sample_name']}" if result.get("sample_name") else ""
    return f"- {target['name']}: {shown} {result.get('unit', '')}{sample}".rstrip()


# Columns of the results table (results/<paper>_extraction.csv)
CSV_COLUMNS = ["item", "parameter", "value", "range", "error", "unit",
               "standard_value", "standard_range", "standard_error", "standard_unit", "conversion_note",
               "sample_name", "status", "method", "evidence", "range_evidence", "error_evidence", "note"]


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
    rows = [["item", "value", "range", "error", "unit", "standard_value", "standard_range",
             "standard_unit", "status"]]
    rows += [[str(r.get(c) if r.get(c) is not None else "") for c in rows[0]] for r in results]
    widths = [min(30, max(len(row[i]) for row in rows)) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell[:30].ljust(w) for cell, w in zip(row, widths)))


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
def plan_targets(prompt: str, params: dict | None = None) -> tuple[list, list]:
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
        msg = chat(messages, tools=PLAN_TOOLS)
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
                params: dict | None = None) -> tuple[dict, list]:
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
    task.append("\nFinish by calling save_result for this item only.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                _paper_message(paper_text),
                {"role": "user", "content": "\n".join(task)}]
    trace = []

    for step in range(1, MAX_STEPS + 1):
        msg = chat(messages, tools=TOOLS)
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

    result = tools.result or {"value": None, "status": "failed",
                              "note": f"No valid result after {MAX_STEPS} steps."}
    result["steps"] = step
    return result, trace


# ============================================================ whole paper
def match_parameters(targets: list[dict], params: dict) -> list[dict]:
    """Attach the matching dictionary entry's name to each item ('' if it is not in the dictionary)."""
    matched = []
    for t in targets:
        match = find_parameter(t["name"], params)
        matched.append({**t, "parameter": match[0] if match else ""})
    return matched


def standardise_results(results: list[dict], params: dict) -> list[dict]:
    """Step 3, after extraction: add the standard_* columns to every result (the originals are kept)."""
    out = []
    for r in results:
        std = standardise(r, standard_unit_for(params.get(r.get("parameter"))))
        print(f"  {r['item']}: {std['conversion_note'] or r.get('status')}")
        out.append({**r, **std})
    return out


def run_agent(paper_text: str, paper_name: str, prompt: str = DEFAULT_PROMPT,
              params: dict | None = None) -> dict:
    params = load_parameters() if params is None else params
    print(f"Request: {prompt}\nParameter dictionary: {len(params)} entries\n\nPlanning...")
    targets, plan_trace = plan_targets(prompt, params)
    targets = match_parameters(targets, params)
    print(f"  {len(targets)} data item(s): " + "; ".join(_describe(t) for t in targets))
    for t in targets:
        if t["parameter"]:
            print(f"    '{t['name']}' -> dictionary entry '{t['parameter']}'")
        else:
            print(f"    '{t['name']}' is not in the dictionary: extracting it anyway, without notes or a "
                  f"range check (add it with: python parameters.py add ...)")

    results, item_traces, done_so_far = [], [], []
    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {_describe(target)}")
        result, trace = extract_one(paper_text, target, prompt, done_so_far, params)
        print(f"  => {_summary(target, result)[2:]}")
        results.append({"item": target["name"], "parameter": target["parameter"],
                        "requested_unit": target.get("unit", ""),
                        "details": target.get("details", ""), **result})
        item_traces.append({"item": target["name"], "trace": trace})
        done_so_far.append(_summary(target, result))

    print("\nStandardising units...")
    results = standardise_results(results, params)

    final = {"paper": paper_name, "model": MODEL, "prompt": prompt, "results": results}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{paper_name}_extraction.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(RESULTS_DIR, f"{paper_name}_extraction.csv"), results)
    with open(os.path.join(RESULTS_DIR, f"{paper_name}_extraction_trace.json"), "w", encoding="utf-8") as f:
        json.dump({"plan": {"targets": targets, "trace": plan_trace}, "items": item_traces},
                  f, indent=2, ensure_ascii=False)
    return final


def load_paper(path: str, si_path: str | None = None) -> tuple[str, str]:
    """Return (text, name). PDFs are converted with pdf_to_text.py and the text saved in text/."""
    name = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".pdf"):
        from pdf_to_text import convert          # needs pymupdf, only imported for PDFs
        text, n_tables = convert(path, si_path)
        print(f"Converted PDF: {n_tables} tables found")
        os.makedirs(TEXT_DIR, exist_ok=True)
        with open(os.path.join(TEXT_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    return text, name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the data asked for in a prompt from one paper.",
                                     epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", help="Paper PDF (or a .txt made by pdf_to_text.py)")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="What data to extract")
    parser.add_argument("--si", help="Supporting Information PDF (optional)")
    args = parser.parse_args()

    text, name = load_paper(args.paper, args.si)
    print(f"Model: {MODEL}\nPaper: {name} ({len(text):,} chars)\n")
    final = run_agent(text, name, args.prompt)
    print("\nFINAL RESULTS:")
    _print_table(final["results"])
    print(f"\nFull results (with evidence quotes): {RESULTS_DIR}/{name}_extraction.json and .csv")
