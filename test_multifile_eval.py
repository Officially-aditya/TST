#!/usr/bin/env python3
"""
TST Multi-File Cross-Dependency Stress Test — Evaluation Script

Runs Qwen3.5-0.8B on each file in the project with two configurations:
  TST:    receives tree context (function signatures + dependency edges from other files)
  Ollama: receives raw file only (no cross-file context)

Both sides get the same prompt template. The ONLY difference is tree context.

Scores 24 planted bugs across 4 difficulty tiers.
Runs N times per side for variance measurement.

Usage:
  source gemma-env/bin/activate
  python test_multifile_eval.py [--runs 1] [--tst-only] [--ollama-only] [--verbose]
  python test_multifile_eval.py --project-dir stress_testing_clean --runs 1
"""

import os
import sys
import json
import time
import subprocess
import argparse
import re
from pathlib import Path
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────
# Bug Registry — ground truth for scoring
# ──────────────────────────────────────────────────────────────

@dataclass
class Bug:
    bug_id: str
    tier: int
    file: str
    function: str
    description: str
    keywords: list[str]  # ALL must appear in response to count as FOUND
    cross_file_from: str = ""  # which other file's knowledge is needed

BUGS = [
    # ── Tier 1: Single-file (no cross-file knowledge needed) ──
    Bug("T1-1", 1, "models/user.py", "validate_email",
        "split('@') can return >2 parts for emails with multiple @",
        ["split", "@", "multiple"]),
    Bug("T1-2", 1, "models/user.py", "record_failed_login",
        "failed_login_count never reset after locking — next failure re-locks immediately",
        ["failed_login_count", "reset"]),
    Bug("T1-3", 1, "models/user.py", "credit",
        "No validation that amount_cents > 0 — negative amounts bypass debit controls",
        ["credit", "negative"]),
    Bug("T1-4", 1, "models/transaction.py", "mark_completed",
        "No status check — can mark failed/reversed transaction as completed",
        ["mark_completed", "status"]),
    Bug("T1-5", 1, "db/connection.py", "ConnectionPool",
        "No close()/shutdown() method — connections leak",
        ["close", "shutdown"]),
    Bug("T1-6", 1, "services/payment.py", "batch_process",
        "Currency() raises ValueError for unknown currencies — aborts entire batch",
        ["Currency", "ValueError"]),

    # ── Tier 2: Signature mismatch (needs callee's signature) ──
    Bug("T2-1", 2, "db/queries.py", "get_user_transactions",
        "3 SQL placeholders but only 2 params — user_id not bound to receiver",
        ["user_id", "receiver", "param"]),
    Bug("T2-2", 2, "services/auth.py", "register",
        "salt from hash_password() not stored — verify_password will always fail",
        ["salt", "store"]),
    Bug("T2-3", 2, "services/auth.py", "change_password",
        "Calls update_user_balance instead of updating password — wrong method entirely",
        ["update_user_balance", "password"]),
    Bug("T2-4", 2, "services/payment.py", "process_payment",
        "user.credit() has no positive check (from user.py) — negative amounts decrease balance",
        ["credit", "negative"]),
    Bug("T2-5", 2, "services/payment.py", "process_payment",
        "Transaction record stores original currency amount, not USD-converted amount",
        ["amount", "usd", "convert"]),
    Bug("T2-6", 2, "services/notification.py", "notify_payment_received",
        "amount_cents displayed raw without converting to dollars",
        ["amount_cents", "dollar"]),

    # ── Tier 3: Semantic contract (needs type + docstring context) ──
    Bug("T3-1", 3, "db/queries.py", "find_user_by_id",
        "pool.acquire() returns None when exhausted — no null check before conn.execute()",
        ["acquire", "None"]),
    Bug("T3-2", 3, "services/auth.py", "login",
        "failed_login_count never reset on successful login — relates to User behavior",
        ["failed_login_count", "reset", "success"]),
    Bug("T3-3", 3, "services/payment.py", "reverse_payment",
        "Searches pending_transactions but completed txns are in DB — always returns None",
        ["pending_transactions", "completed"]),
    Bug("T3-4", 3, "services/notification.py", "notify_low_balance",
        "Comparison inverted — returns False when balance IS low (> instead of <)",
        ["balance", "threshold", "invert"]),

    # ── Tier 4: System-level (needs full dependency graph understanding) ──
    Bug("T4-1", 4, "services/payment.py", "process_payment",
        "Sender debited before receiver existence check — funds vanish if receiver not found",
        ["debit", "receiver", "rollback"]),
    Bug("T4-2", 4, "services/payment.py", "reverse_payment",
        "mark_completed has no status guard (T1-4) — reversed txn can be re-completed",
        ["mark_completed", "status", "reverse"]),
    Bug("T4-3", 4, "services/notification.py", "send_batch_summary",
        "Depends on payment.py error format being list[str] — coupling bug",
        ["errors", "format", "coupling"]),
    Bug("T4-4", 4, "app.py", "run_demo",
        "process_payment returns original EUR amount but balances updated in USD",
        ["EUR", "USD", "amount", "mismatch"]),
    Bug("T4-5", 4, "app.py", "run_demo",
        "Notification uses 1000 (EUR cents) but balance changed by 1080 (USD cents)",
        ["1000", "1080"]),
    Bug("T4-6", 4, "app.py", "run_demo",
        "Connection pool never closed — resource leak across app lifecycle",
        ["pool", "close", "leak"]),
]


# ──────────────────────────────────────────────────────────────
# File parser — extracts signatures for tree context
# ──────────────────────────────────────────────────────────────

def parse_python_signatures(filepath: str) -> list[str]:
    """Extract function and class signatures from a Python file."""
    signatures = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            sig = stripped
            j = i + 1
            while j < len(lines) and ")" not in sig:
                sig += " " + lines[j].strip()
                j += 1
            if ":" in sig:
                # Find the colon that closes the def — must be after the closing paren,
                # not the colon inside type annotations like "password: str"
                close_paren = sig.rfind(")")
                colon_pos = sig.index(":", close_paren if close_paren >= 0 else 0)
                sig = sig[:colon_pos + 1]
            signatures.append(sig)
        elif stripped.startswith("class "):
            sig = stripped
            if ":" in sig:
                sig = sig[:sig.index(":") + 1]
            signatures.append(sig)
    return signatures


def parse_stm_evidence(filepath: str) -> list[dict]:
    """
    Extract STM-style Preference entries from a file's function signatures.
    Returns compact key: value (weight) evidence tuples — much denser than raw signatures.

    Evidence types generated:
      {func}_returns: <type>          — return type contract
      {func}_can_return_none: True    — explicit None-safety signal (from Optional returns)
      {func}_arg_{name}: <type>       — parameter type contract
      {func}_exists: True             — method name existence (for wrong-method-name bugs)
      {class}_exists: True            — class existence
    """
    evidence = []
    sigs = parse_python_signatures(filepath)

    for sig in sigs:
        # Class definition
        if sig.startswith("class "):
            m = re.match(r"class (\w+)", sig)
            if m and not m.group(1).startswith("_"):
                evidence.append({"key": f"{m.group(1)}_exists", "value": "True", "weight": 0.82})
            continue

        # Function definition
        m = re.match(r"(?:async )?def (\w+)\(([^)]*)\)(?:\s*->\s*(.+?))?:?$", sig.strip())
        if not m:
            continue

        fname, params_raw, ret_raw = m.group(1), m.group(2).strip(), m.group(3)

        if fname.startswith("__"):
            continue

        # Return type evidence
        if ret_raw:
            rt = ret_raw.strip().rstrip(":")
            is_optional = "Optional" in rt or ("None" in rt and rt != "None")
            weight = 0.88 if is_optional else 0.95
            evidence.append({"key": f"{fname}_returns", "value": rt, "weight": weight})
            if is_optional:
                evidence.append({"key": f"{fname}_can_return_none", "value": "True", "weight": 0.90})

        # Parameter type evidence
        for param in params_raw.split(","):
            param = param.strip()
            if not param or param in ("self", "cls") or param.startswith("*"):
                continue
            if ":" in param:
                pname, ptype = param.split(":", 1)
                pname = pname.strip()
                if "=" in ptype:
                    ptype = ptype.split("=")[0]
                ptype = ptype.strip()
                if ptype and ptype not in ("Any", "object"):
                    evidence.append({"key": f"{fname}_arg_{pname}", "value": ptype, "weight": 0.92})

        # Existence entry — lets model detect wrong-method-name calls (T2-3 class)
        evidence.append({"key": f"{fname}_exists", "value": "True", "weight": 0.82})

    return evidence


def build_stm_context(project_dir: str, current_file: str, file_content: str = "") -> str:
    """
    Build compact STM evidence context from directly imported files.
    Uses Preference-style key: value (weight) format instead of raw signatures.
    Each entry is a discrete learned fact — denser and more targeted than tree context.
    Caps at 35 evidence entries (~200 tokens max).
    """
    if not file_content:
        return ""

    imported = get_direct_imports(file_content, project_dir, current_file)
    if not imported:
        return ""

    context_parts = []
    evidence_count = 0
    MAX_EVIDENCE = 35

    for rel_path in imported:
        file_path = os.path.join(project_dir, rel_path)
        if not os.path.exists(file_path):
            continue

        evidence = parse_stm_evidence(file_path)
        if not evidence:
            continue

        file_parts = [f"\nFrom {rel_path}:"]
        for e in evidence:
            if evidence_count >= MAX_EVIDENCE:
                break
            file_parts.append(f"  {e['key']}: {e['value']} (w={e['weight']:.2f})")
            evidence_count += 1

        if len(file_parts) > 1:
            context_parts.extend(file_parts)

        if evidence_count >= MAX_EVIDENCE:
            break

    if not context_parts:
        return ""

    header = (
        "\n[STM MEMORY — weighted cross-file evidence from direct imports]\n"
        "Each entry is a learned fact about a function in an imported file.\n"
        "Use these to detect type mismatches, wrong methods, None-safety violations, and contract bugs.\n"
    )
    return header + "\n".join(context_parts) + "\n[END STM MEMORY]\n"


def get_direct_imports(file_content: str, project_dir: str, current_file: str) -> list[str]:
    """
    Parse import statements in file_content and resolve to relative file paths
    within the project directory. Only returns files that actually exist in the project.
    """
    project = Path(project_dir)
    seen = {}  # rel_path -> True, preserves insertion order

    for line in file_content.splitlines():
        line = line.strip()
        m = re.match(r'^from ([\w.]+) import', line) or re.match(r'^import ([\w.]+)', line)
        if not m:
            continue
        module = m.group(1)
        rel_path = module.replace(".", "/") + ".py"
        abs_path = project / rel_path
        if abs_path.exists() and rel_path != current_file:
            seen[rel_path] = True

    return list(seen.keys())


def build_tree_context(project_dir: str, current_file: str, file_content: str = "") -> str:
    """
    Build tree context string: signatures ONLY from files the current file imports.
    Caps at 20 signatures (~300 tokens) to stay within 0.8B attention window.
    """
    if not file_content:
        return ""

    imported = get_direct_imports(file_content, project_dir, current_file)
    if not imported:
        return ""

    context_parts = []
    sig_count = 0
    MAX_SIGS = 20

    for rel_path in imported:
        file_path = os.path.join(project_dir, rel_path)
        sigs = parse_python_signatures(file_path)
        if not sigs:
            continue

        context_parts.append(f"\n--- {rel_path} exports ---")
        for sig in sigs:
            if sig_count >= MAX_SIGS:
                break
            context_parts.append(f"  {sig}")
            sig_count += 1

        if sig_count >= MAX_SIGS:
            break

    if not context_parts:
        return ""

    header = (
        "\n[PROJECT CONTEXT — function signatures from files this module imports]\n"
        "Use these to detect cross-file type mismatches, wrong argument counts,\n"
        "contract violations, and dependency issues.\n"
    )
    return header + "\n".join(context_parts) + "\n[END PROJECT CONTEXT]\n"


# ──────────────────────────────────────────────────────────────
# Model runners
# ──────────────────────────────────────────────────────────────

def run_tst_analysis(file_content: str, tree_context: str, model, tokenizer, device) -> tuple[str, float]:
    """Run Qwen3.5-0.8B with tree context via HuggingFace."""
    import torch

    system_msg = (
        "You are a code reviewer. Analyze the provided code and find ALL bugs, errors, "
        "and issues. For each bug: state the function name, what is wrong, and why it "
        "is a problem. Check for: missing validations, wrong parameter counts, type "
        "mismatches with imported functions, contract violations, resource leaks, and "
        "logic errors. Be specific and exhaustive."
    )

    if tree_context:
        user_msg = f"{tree_context}\nAnalyze this file for bugs. Use the STM evidence above to detect cross-file issues:\n\n{file_content}"
    else:
        user_msg = f"Analyze this file for bugs:\n\n{file_content}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    elapsed_ms = (time.time() - start) * 1000

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response, elapsed_ms


def run_ollama_analysis(file_content: str, model_name: str = "qwen3.5:0.8b") -> tuple[str, float]:
    """Run same model via Ollama (no tree context)."""
    system_msg = (
        "You are a code reviewer. Analyze the provided code and find ALL bugs, errors, "
        "and issues. For each bug: state the function name, what is wrong, and why it "
        "is a problem. Check for: missing validations, wrong parameter counts, type "
        "mismatches with imported functions, contract violations, resource leaks, and "
        "logic errors. Be specific and exhaustive."
    )

    user_msg = f"Analyze this file for bugs:\n\n{file_content}"
    full_prompt = f"[SYSTEM] {system_msg}\n\n[USER] {user_msg}"

    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed_ms = (time.time() - start) * 1000
        return result.stdout.strip(), elapsed_ms
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.time() - start) * 1000
        return "[TIMEOUT after 600s]", elapsed_ms
    except FileNotFoundError:
        return "[Ollama not found]", 0


# ──────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────

def score_response(response: str, bugs_in_file: list[Bug]) -> dict:
    """Score a response against the bug registry for a specific file."""
    response_lower = response.lower()
    results = {}

    for bug in bugs_in_file:
        all_found = all(kw.lower() in response_lower for kw in bug.keywords)
        if all_found:
            results[bug.bug_id] = "FOUND"
        else:
            found_count = sum(1 for kw in bug.keywords if kw.lower() in response_lower)
            if found_count >= len(bug.keywords) / 2:
                results[bug.bug_id] = "PARTIAL"
            else:
                results[bug.bug_id] = "MISSED"

    return results


def count_hallucinations(response: str, file_path: str) -> int:
    """Rough hallucination detector: count claims about functions not in the file."""
    mentioned = set(re.findall(r'`?(\w+)\(\)`?', response))

    actual = set()
    with open(file_path, "r") as f:
        for line in f:
            m = re.match(r'\s*def (\w+)\(', line)
            if m:
                actual.add(m.group(1))
            m = re.match(r'\s*class (\w+)', line)
            if m:
                actual.add(m.group(1))

    builtins = {"print", "len", "str", "int", "float", "list", "dict", "set",
                "range", "isinstance", "type", "max", "min", "sum", "any", "all",
                "open", "None", "True", "False", "self", "return", "raise"}

    phantom = mentioned - actual - builtins
    return len([f for f in phantom if len(f) > 3 and f[0].islower()])


# ──────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TST Multi-File Stress Test")
    parser.add_argument("--runs", type=int, default=1, help="Runs per side per file")
    parser.add_argument("--tst-only", action="store_true")
    parser.add_argument("--ollama-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--project-dir", default="stress_testing_clean")
    args = parser.parse_args()

    project_dir = args.project_dir
    if not os.path.isdir(project_dir):
        print(f"ERROR: {project_dir}/ not found.")
        sys.exit(1)

    test_files = [
        "models/user.py",
        "models/transaction.py",
        "utils/validators.py",
        "utils/crypto.py",
        "db/connection.py",
        "db/queries.py",
        "services/auth.py",
        "services/payment.py",
        "services/notification.py",
        "app.py",
    ]

    # Load TST model
    tst_model, tst_tokenizer, device = None, None, None
    if not args.ollama_only:
        print("Loading Qwen3.5-0.8B for TST side...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen3.5-0.8B"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        tst_tokenizer = AutoTokenizer.from_pretrained(model_id)
        tst_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        tst_model.eval()
        print(f"  Loaded on {device}\n")

    # ── Run evaluations ──
    all_results = {"tst": {}, "ollama": {}}

    for file_rel in test_files:
        file_path = os.path.join(project_dir, file_rel)
        if not os.path.exists(file_path):
            print(f"  SKIP: {file_rel} not found")
            continue

        with open(file_path, "r") as f:
            file_content = f.read()

        file_bugs = [b for b in BUGS if b.file == file_rel]
        if not file_bugs:
            continue

        print(f"\n{'='*60}")
        print(f"  Analyzing: {file_rel} ({len(file_bugs)} bugs, tiers {sorted(set(b.tier for b in file_bugs))})")
        print(f"{'='*60}")

        tree_context = build_stm_context(project_dir, file_rel, file_content)

        # ── TST runs ──
        if not args.ollama_only:
            tst_file_results = []
            for run in range(args.runs):
                print(f"  TST run {run+1}/{args.runs}...", end=" ", flush=True)
                response, latency = run_tst_analysis(
                    file_content, tree_context, tst_model, tst_tokenizer, device
                )
                scores = score_response(response, file_bugs)
                hallucinations = count_hallucinations(response, file_path)
                tst_file_results.append({
                    "run": run + 1,
                    "scores": scores,
                    "hallucinations": hallucinations,
                    "latency_ms": latency,
                    "response_length": len(response),
                    "response": response if args.verbose else "",
                })
                found = sum(1 for v in scores.values() if v == "FOUND")
                partial = sum(1 for v in scores.values() if v == "PARTIAL")
                print(f"{found}/{len(file_bugs)} found, {partial} partial, "
                      f"{hallucinations} halluc, {latency:.0f}ms")
                if args.verbose:
                    for bug_id, status in scores.items():
                        print(f"    {bug_id}: {status}")

            all_results["tst"][file_rel] = tst_file_results

        # ── Ollama runs ──
        if not args.tst_only:
            ollama_file_results = []
            for run in range(args.runs):
                print(f"  Ollama run {run+1}/{args.runs}...", end=" ", flush=True)
                response, latency = run_ollama_analysis(file_content)
                scores = score_response(response, file_bugs)
                hallucinations = count_hallucinations(response, file_path)
                ollama_file_results.append({
                    "run": run + 1,
                    "scores": scores,
                    "hallucinations": hallucinations,
                    "latency_ms": latency,
                    "response_length": len(response),
                    "response": response if args.verbose else "",
                })
                found = sum(1 for v in scores.values() if v == "FOUND")
                partial = sum(1 for v in scores.values() if v == "PARTIAL")
                print(f"{found}/{len(file_bugs)} found, {partial} partial, "
                      f"{hallucinations} halluc, {latency:.0f}ms")
                if args.verbose:
                    for bug_id, status in scores.items():
                        print(f"    {bug_id}: {status}")

            all_results["ollama"][file_rel] = ollama_file_results

    # ── Aggregate and report ──
    print(f"\n{'='*70}")
    print("  FINAL RESULTS — AGGREGATED ACROSS ALL RUNS")
    print(f"{'='*70}")

    for side in ["tst", "ollama"]:
        if not all_results[side]:
            continue

        print(f"\n  ── {side.upper()} ──")
        tier_scores = {1: [], 2: [], 3: [], 4: []}
        total_hallucinations = 0
        total_latency = 0
        run_count = 0

        for file_rel, runs in all_results[side].items():
            for run_data in runs:
                run_count += 1
                total_latency += run_data["latency_ms"]
                total_hallucinations += run_data["hallucinations"]
                for bug in BUGS:
                    if bug.file == file_rel:
                        status = run_data["scores"].get(bug.bug_id, "MISSED")
                        tier_scores[bug.tier].append(1 if status == "FOUND" else 0)

        print(f"\n  {'Tier':<8} {'Found Rate':<15} {'Description'}")
        print(f"  {'─'*60}")
        for tier in [1, 2, 3, 4]:
            scores = tier_scores[tier]
            if scores:
                rate = sum(scores) / len(scores) * 100
                labels = {
                    1: "Single-file (control)",
                    2: "Signature mismatch",
                    3: "Semantic contract",
                    4: "System-level",
                }
                print(f"  Tier {tier:<4} {rate:>5.1f}%          {labels[tier]}")

        all_scores = [s for tier in tier_scores.values() for s in tier]
        overall = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
        avg_latency = total_latency / run_count if run_count else 0
        avg_halluc = total_hallucinations / run_count if run_count else 0

        print(f"\n  Overall:     {overall:.1f}%")
        print(f"  Avg latency: {avg_latency:.0f} ms")
        print(f"  Avg halluc:  {avg_halluc:.1f} per file")

    # ── Save raw results ──
    output_path = "multifile_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Raw results saved to {output_path}")

    # ── Expected outcome comparison ──
    print(f"\n{'='*70}")
    print("  EXPECTED vs ACTUAL (hypothesis check)")
    print(f"{'='*70}")
    print("  Tier 1: Both ~80-100%  (baseline — no cross-file knowledge needed)")
    print("  Tier 2: TST >> Ollama  (signature context is the key differentiator)")
    print("  Tier 3: TST > Ollama   (contract knowledge helps, 0.8B may still miss)")
    print("  Tier 4: Both low       (system-level likely exceeds 0.8B capacity)")
    print("\n  If Tier 2 TST >> Ollama, Tree Memory is validated.")
    print("  If Tier 4 both low, the 0.8B ceiling is documented.")


if __name__ == "__main__":
    main()
