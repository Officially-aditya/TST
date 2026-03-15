"""
router_eval.py

Standalone accuracy + latency evaluation for the TST Memory Router.

Loads models directly (no HTTP server needed) and runs 4 canonical test
queries through the same tiered inference pipeline used in router/server.py.
Also runs the old Layer 2 approach (Qwen-only, regex) for direct comparison.

Run:
  source gemma-env/bin/activate
  python router_eval.py
"""

from __future__ import annotations

import json
import re
import time
import sys
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from router.tools import TOOL_SCHEMAS, FG_TOOL_SCHEMAS, VALID_ROUTES

# ─── Config ──────────────────────────────────────────────────────────────────

FG_MODEL_ID = "google/functiongemma-270m-it"
Q3_MODEL_ID = "Qwen/Qwen3.5-0.8B"

ROUTE_TOKENS = ["route_to_stm", "route_to_ltm", "route_to_tree", "route_to_cloud"]

# 4 canonical test queries — one per expected route
TEST_CASES = [
    {"query": "What did we just discuss?",                    "payload": "Recent chat about memory routing.", "expected": "route_to_stm"},
    {"query": "User always prefers TypeScript over JavaScript","payload": "TypeScript preference rule.",        "expected": "route_to_ltm"},
    {"query": "Fix the syntax error on line 53 of main.rs",  "payload": "",                                  "expected": "route_to_tree"},
    {"query": "What is the capital of France?",              "payload": "",                                  "expected": "route_to_cloud"},
]

# FunctionGemma: must use role "developer" + this exact phrase; tools via apply_chat_template(tools=)
_FG_DEVELOPER_MSG = "You are a model that can do function calling with the following functions"

# Qwen3: JSON tool-call prompt embedded in system message
_Q3_SYSTEM_PROMPT = (
    "You are a memory routing engine. "
    "Given a user query, call exactly one of the provided tools. "
    'Respond ONLY with a valid JSON object in this exact format:\n'
    '{"tool_calls": [{"name": "<tool_name>", "args": {<args>}}]}\n'
    "Do not include any explanation or text outside the JSON."
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {msg}")
    print(f"{'-' * 60}")

def get_device() -> str:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_fg(model_id: str, device: str):
    """Load FunctionGemma with AutoProcessor (required for correct template handling).
    MUST use float32 — FunctionGemma's RecurrentGemma SSM layers produce NaN/pad
    tokens under float16 on any device (confirmed via diagnostic on 2026-03-14)."""
    print(f"  Loading {model_id} (AutoProcessor, float32) ...")
    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(model_id)
    mdl  = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    mdl.eval()
    print(f"  Loaded in {(time.perf_counter()-t0)*1000:.0f} ms")
    return proc, mdl

def load_q3(model_id: str, device: str):
    """Load Qwen3.5 (multimodal model) with correct class."""
    print(f"  Loading {model_id} (AutoTokenizer) ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    from transformers import AutoModelForImageTextToText
    mdl = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float16).to(device)
    mdl.eval()
    print(f"  Loaded in {(time.perf_counter()-t0)*1000:.0f} ms")
    return tok, mdl

# ── FunctionGemma prompt + parser ────────────────────────────────────────────

def build_fg_prompt(processor, query: str, payload: str):
    """
    Correct FunctionGemma format:
      role="developer" + activation phrase + tools via apply_chat_template(tools=FG_TOOL_SCHEMAS)
    """
    user_content = f"Query: '{query}'"
    if payload:
        user_content += f"\nPayload: '{payload}'"
    messages = [
        {"role": "developer", "content": _FG_DEVELOPER_MSG},
        {"role": "user",      "content": user_content},
    ]
    return processor.apply_chat_template(
        messages, tools=FG_TOOL_SCHEMAS,
        add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )

def parse_fg_output(raw: str) -> str | None:
    """
    Parse FunctionGemma output: <start_function_call>call:name{p:<escape>v<escape>}<end_function_call>
    Falls back to substring scan.
    """
    name_match = re.search(r'call:(\w+)\{', raw)
    if name_match:
        name = name_match.group(1)
        if name in VALID_ROUTES:
            return name
    for route in ROUTE_TOKENS:
        if route in raw:
            return route
    return None

def infer_fg(processor, model, query: str, payload: str, device: str):
    encoded = build_fg_prompt(processor, query, payload)
    # Move ALL encoded tensors to device (input_ids + attention_mask).
    # Official HF example uses **inputs.to(device) — without attention_mask
    # the model generates only <pad> tokens and produces no output.
    encoded_on_device = {k: v.to(device) for k, v in encoded.items()}
    n_in = encoded_on_device["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **encoded_on_device,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=processor.eos_token_id,
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    new_ids = out[0][n_in:]
    # Keep special tokens so <start_function_call> is visible to the parser
    raw = processor.decode(new_ids, skip_special_tokens=False)
    return raw, n_in, new_ids.shape[0], wall_ms

# ── Qwen3 prompt + parser ─────────────────────────────────────────────────────

def build_q3_prompt(tokenizer, query: str, payload: str):
    tools_json = json.dumps(TOOL_SCHEMAS, indent=2)
    system = f"{_Q3_SYSTEM_PROMPT}\n\nAvailable tools:\n{tools_json}"
    user = f"Query: '{query}'"
    if payload:
        user += f"\nPayload: '{payload}'"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )

def parse_q3_output(raw: str) -> str | None:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        calls = obj.get("tool_calls", [])
        if calls and calls[0].get("name") in VALID_ROUTES:
            return calls[0]["name"]
    except Exception:
        pass
    match = re.search(r'\{[\s\S]*?"tool_calls"[\s\S]*?\}', raw)
    if match:
        try:
            obj = json.loads(match.group())
            calls = obj.get("tool_calls", [])
            if calls and calls[0].get("name") in VALID_ROUTES:
                return calls[0]["name"]
        except Exception:
            pass
    for route in ROUTE_TOKENS:
        if route in raw:
            return route
    return None

def infer_q3(tokenizer, model, query: str, payload: str, device: str):
    encoded   = build_q3_prompt(tokenizer, query, payload)
    input_ids = encoded["input_ids"].to(device)
    n_in      = input_ids.shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    new_ids = out[0][n_in:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    return raw, n_in, new_ids.shape[0], wall_ms

# ─── Old Layer 2 approach (baseline) ─────────────────────────────────────────

def run_layer2_baseline(q3_tok, q3_mdl, device: str) -> list[dict]:
    """Replicate layer2_tests.py: Qwen-only, manual chat template, regex extraction."""
    results = []
    for tc in TEST_CASES:
        router_prompt = (
            "<|im_start|>system\n"
            "You are a routing classification engine. You must map the user query to one of these exact terms: 'STM', 'LTM', 'Tree', 'Cloud'.\n"
            "Here is the routing logic:\n"
            "- If the query asks about 'recent discussion' or 'what we just did', route to STM.\n"
            "- If the query states a preference or long-lasting rule (e.g. 'always use X'), route to LTM.\n"
            "- If the query asks to fix code, analyze syntax, or search a file structure, route to Tree.\n"
            "- If the query asks for general world knowledge (e.g. Weather, history), route to Cloud.\n"
            "You MUST output exactly ONE word.\n<|im_end|>\n"
            f"<|im_start|>user\nQuery: '{tc['query']}'<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = q3_tok(router_prompt, return_tensors="pt").to(device)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = q3_mdl.generate(
                **inputs, max_new_tokens=100,
                do_sample=False,
                pad_token_id=q3_tok.eos_token_id,
            )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        raw = q3_tok.decode(out[0][n_in:], skip_special_tokens=True).strip()

        # Old regex approach
        matched = None
        if "STM" in raw:   matched = "STM"
        elif "LTM" in raw: matched = "LTM"
        elif "Tree" in raw:matched = "Tree"
        elif "Cloud" in raw:matched = "Cloud"

        # Map expected to old label format
        expected_short = tc["expected"].replace("route_to_", "").upper()
        if expected_short == "TREE": expected_short = "Tree"
        if expected_short == "CLOUD": expected_short = "Cloud"

        passed = matched == expected_short if matched else False
        results.append({
            "query":    tc["query"],
            "expected": expected_short,
            "got":      matched or f"NONE (raw: {raw[:60]})",
            "passed":   passed,
            "wall_ms":  round(wall_ms, 1),
            "raw":      raw[:80],
        })
    return results

# ─── New tiered router ────────────────────────────────────────────────────────

def run_new_router(fg_proc, fg_mdl, q3_tok, q3_mdl, device: str) -> list[dict]:
    results = []
    for tc in TEST_CASES:
        # Tier 1 — FunctionGemma (correct format: developer role + tools= param)
        raw_fg, n_in_fg, n_new_fg, t_fg = infer_fg(fg_proc, fg_mdl, tc["query"], tc["payload"], device)
        route = parse_fg_output(raw_fg)
        tier = "FunctionGemma-270M"
        wall_ms = t_fg

        # Tier 2 fallback — Qwen (JSON prompt format)
        if route is None:
            raw_q3, n_in_q3, n_new_q3, t_q3 = infer_q3(q3_tok, q3_mdl, tc["query"], tc["payload"], device)
            route = parse_q3_output(raw_q3)
            tier = "Qwen3.5-0.8B (fallback)"
            wall_ms = t_fg + t_q3

        # Default
        if route is None:
            route = "route_to_cloud"
            tier = "default-fallback"

        passed = (route == tc["expected"])
        results.append({
            "query":    tc["query"],
            "expected": tc["expected"],
            "got":      route,
            "passed":   passed,
            "tier":     tier,
            "wall_ms":  round(wall_ms, 1),
            "fg_raw":   raw_fg[:120],
        })
    return results

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"\nDevice: {device}")

    banner("Loading Models")
    fg_proc, fg_mdl = load_fg(FG_MODEL_ID, device)
    q3_tok,  q3_mdl = load_q3(Q3_MODEL_ID, device)

    # ── Baseline: Old Layer 2 (Qwen-only regex) ───────────────────────────
    banner("BASELINE  -- Layer 2 approach (Qwen-only, regex extraction)")
    baseline = run_layer2_baseline(q3_tok, q3_mdl, device)
    b_pass = sum(1 for r in baseline if r["passed"])
    b_avg  = sum(r["wall_ms"] for r in baseline) / len(baseline)
    for r in baseline:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['query'][:45]:<45}  expected={r['expected']:<6}  got={str(r['got']):<20}  {r['wall_ms']} ms")
    print(f"\n  Baseline accuracy : {b_pass}/{len(baseline)}  ({b_pass/len(baseline)*100:.0f}%)")
    print(f"  Baseline avg lat  : {b_avg:.0f} ms")

    # ── New Router: Tiered FG → Q3 ────────────────────────────────────────
    banner("NEW ROUTER -- Tiered (FunctionGemma-270M -> Qwen3.5-0.8B fallback)")
    new_res = run_new_router(fg_proc, fg_mdl, q3_tok, q3_mdl, device)
    n_pass = sum(1 for r in new_res if r["passed"])
    n_avg  = sum(r["wall_ms"] for r in new_res) / len(new_res)
    for r in new_res:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['query'][:45]:<45}  expected={r['expected']:<15}  got={r['got']:<20}  tier={r['tier']}  {r['wall_ms']} ms")
        print(f"         FG raw: {r['fg_raw']}")
    print(f"\n  New router accuracy : {n_pass}/{len(new_res)}  ({n_pass/len(new_res)*100:.0f}%)")
    print(f"  New router avg lat  : {n_avg:.0f} ms")

    # ── Delta Summary ─────────────────────────────────────────────────────
    banner("DELTA SUMMARY")
    acc_delta = (n_pass - b_pass)
    lat_delta = (n_avg - b_avg)
    print(f"  Accuracy : {b_pass}/{len(baseline)} -> {n_pass}/{len(new_res)}   ({'+' if acc_delta >= 0 else ''}{acc_delta} correct)")
    print(f"  Avg lat  : {b_avg:.0f} ms -> {n_avg:.0f} ms   ({'+' if lat_delta >= 0 else ''}{lat_delta:.0f} ms)")
    print()

    # Save results
    output = {
        "baseline_layer2": baseline,
        "new_router":      new_res,
        "summary": {
            "baseline_accuracy": f"{b_pass}/{len(baseline)}",
            "new_accuracy":      f"{n_pass}/{len(new_res)}",
            "baseline_avg_ms":   round(b_avg, 1),
            "new_avg_ms":        round(n_avg, 1),
        }
    }
    import pathlib
    out_path = pathlib.Path("router/eval_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved -> {out_path}")

if __name__ == "__main__":
    main()
