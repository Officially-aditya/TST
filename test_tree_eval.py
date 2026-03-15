#!/usr/bin/env python3
"""
test_tree_eval_clean.py — Clean Cross-File Reasoning Test

Design:
  - Both TST and Ollama receive ONLY file1.py (no file2.py code shown)
  - TST gets tree context: function signatures parsed from all project files,
    including file2.py::function_e(n: int) -> float
  - Ollama gets file1.py raw — blind to file2's function signatures

Planted bugs in file1.py (no BUG comments):
  A. function_a: counts[item] = 0  (first occurrence should be 1)
  B. function_b: trimmed = scores[1:] then /len(trimmed)  (ZeroDivisionError when len==1)
  C. function_c: function_e(str(rec["score"]))  (type mismatch — function_e expects int)

Bug C is the cross-file discriminator:
  - File1 only shows the call: function_e(str(rec["score"]))
  - To know this is wrong, you need file2's signature: function_e(n: int)
  - TST injects this via tree context; Ollama has no such information

Expected outcome:
  - Both sides find A and B (visible in file1 alone)
  - Only TST finds C (requires knowledge of function_e's signature from file2)
"""

import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from cli import parse_file, _WORKER_CODE_SYSTEM

# ─── Config ───────────────────────────────────────────────────────────────────

Q3_MODEL_ID  = "Qwen/Qwen3.5-0.8B"
PROJECT_DIR  = Path(__file__).parent / "test_project_clean"
SEP          = "─" * 64

# Bugs scored against file1.py only (both sides are given file1.py)
BUGS = {
    "A_counts_off_by_one": {
        "desc": "function_a: counts[item] = 0 on first occurrence (should be 1)",
        "keys": ["0", "count", "function_a"],
        "cross_file": False,
    },
    "B_zerodivision": {
        "desc": "function_b: ZeroDivisionError when len(scores) == 1 (trimmed is empty)",
        "keys": ["division", "trimmed", "empty"],
        "cross_file": False,
    },
    "C_str_to_function_e": {
        "desc": "function_c: str(rec['score']) passed to function_e which expects int -> TypeError",
        "keys": ["str", "function_e", "int"],
        "cross_file": True,   # <- only TST can detect this via tree context
    },
}


# ─── Model loading ────────────────────────────────────────────────────────────

def load_qwen():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading {Q3_MODEL_ID} on {device} ...")
    t0  = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(Q3_MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        Q3_MODEL_ID, dtype=torch.float16
    ).to(device)
    mdl.eval()
    print(f"  Loaded in {(time.perf_counter() - t0) * 1000:.0f} ms\n")
    return tok, mdl, device


# ─── Inference helpers ────────────────────────────────────────────────────────

def infer_hf(tok, mdl, device, system: str, user: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )
    ids = enc["input_ids"].to(device)
    t0  = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(
            ids,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    ms       = (time.perf_counter() - t0) * 1000
    response = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return response, ms


def infer_ollama(prompt: str) -> tuple[str, float]:
    t0     = time.perf_counter()
    result = subprocess.run(
        ["ollama", "run", "qwen3.5:0.8b"],
        input=prompt,
        capture_output=True, text=True, timeout=600,
    )
    ms = (time.perf_counter() - t0) * 1000
    return result.stdout.strip(), ms


# ─── Tree context builder ─────────────────────────────────────────────────────

def build_tree_context_for_file1(project_dir: Path) -> str:
    """
    Parse ALL files in the project. Return a tree context string that tells
    the model about function signatures available in other files.
    Only function_e's signature (n: int -> float) is the discriminating signal.
    """
    all_files = sorted(project_dir.glob("file*.py"))
    file1     = project_dir / "file1.py"

    context_parts = []
    for f in all_files:
        if f == file1:
            continue
        parsed = parse_file(str(f))
        if parsed["functions"]:
            # Include raw source of function signatures from other files
            src = f.read_text()
            sigs = []
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("def "):
                    sigs.append(stripped.rstrip(":"))
            if sigs:
                context_parts.append(
                    f"{f.name} exports: " + " | ".join(sigs)
                )

    if context_parts:
        return "Other files in this project:\n" + "\n".join(
            f"  {c}" for c in context_parts
        )
    return ""


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_response(response: str) -> dict[str, bool]:
    r_low  = response.lower()
    return {
        bug_id: all(kw in r_low for kw in info["keys"])
        for bug_id, info in BUGS.items()
    }


def print_score(label: str, found: dict[str, bool]):
    total = len(BUGS)
    n     = sum(found.values())
    print(f"\n  {label}  {n}/{total} bugs found:")
    for bug_id, hit in found.items():
        info  = BUGS[bug_id]
        cross = "  [CROSS-FILE]" if info["cross_file"] else ""
        tick  = "✓" if hit else "✗"
        print(f"    {tick} {bug_id}{cross}")
        if hit:
            print(f"      → {info['desc']}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    file1 = PROJECT_DIR / "file1.py"
    if not file1.exists():
        print(f"[ERROR] {file1} not found")
        return

    file1_code = file1.read_text()
    tree_ctx   = build_tree_context_for_file1(PROJECT_DIR)

    print(SEP)
    print("Clean Cross-File Reasoning Test")
    print(SEP)
    print(f"\nCode given to both models: file1.py ({len(file1_code.splitlines())} lines)")
    print(f"file2.py: NOT shown as code to either model")
    print(f"\nTree context injected into TST only:")
    print(tree_ctx if tree_ctx else "  (none)")
    print()

    tok, mdl, device = load_qwen()

    # ── TST ───────────────────────────────────────────────────────────────────
    print(SEP)
    print("TST — Qwen3.5-0.8B (MPS) | _WORKER_CODE_SYSTEM + tree context")
    print(SEP)

    tst_system = _WORKER_CODE_SYSTEM
    if tree_ctx:
        tst_system += f"\n\n{tree_ctx}"

    tst_user = "Find all bugs and errors in this code:\n\n" + file1_code
    tst_resp, tst_ms = infer_hf(tok, mdl, device, tst_system, tst_user)

    print(f"\n[{tst_ms:.0f} ms]\n")
    print(tst_resp)
    tst_found = score_response(tst_resp)
    print_score("TST", tst_found)

    # ── Ollama ────────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("Ollama — qwen3.5:0.8b (CPU) | plain prompt, no tree context")
    print(SEP)

    ollama_prompt = "Find all bugs and errors in this code:\n\n" + file1_code
    ollama_resp, ollama_ms = infer_ollama(ollama_prompt)

    print(f"\n[{ollama_ms:.0f} ms]\n")
    print(ollama_resp)
    ollama_found = score_response(ollama_resp)
    print_score("Ollama", ollama_found)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("SUMMARY")
    print(SEP)
    print(f"\n  {'Bug':<30}  TST   Ollama   cross-file")
    print(f"  {'─'*30}  ─────  ──────   ──────────")
    for bug_id, info in BUGS.items():
        t = "✓" if tst_found[bug_id]    else "✗"
        o = "✓" if ollama_found[bug_id] else "✗"
        c = "YES" if info["cross_file"] else "no"
        print(f"  {bug_id:<30}   {t}     {o}      {c}")

    tst_total    = sum(tst_found.values())
    ollama_total = sum(ollama_found.values())
    total        = len(BUGS)
    print(f"\n  TST  : {tst_total}/{total}")
    print(f"  Ollama: {ollama_total}/{total}")
    print(f"\n  Timing:  TST {tst_ms:.0f} ms | Ollama {ollama_ms:.0f} ms")

    cross_id = "C_str_to_function_e"
    tst_got_cross    = tst_found[cross_id]
    ollama_got_cross = ollama_found[cross_id]
    print(f"\n  Cross-file bug (C):")
    print(f"    TST caught it   : {'YES' if tst_got_cross    else 'NO'}")
    print(f"    Ollama caught it: {'YES' if ollama_got_cross else 'NO'}")
    if tst_got_cross and not ollama_got_cross:
        print(f"\n  ✓ TST advantage CONFIRMED — tree context enabled cross-file detection")
    elif not tst_got_cross and not ollama_got_cross:
        print(f"\n  — Neither detected the cross-file bug (model too small?)")
    elif ollama_got_cross:
        print(f"\n  — Ollama also detected it (guessed from str() + function call pattern)")
    print(SEP)


if __name__ == "__main__":
    run()
