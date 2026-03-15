#!/usr/bin/env python3
"""
Code reasoning validation — clean re-run.
  TST side  : Qwen3.5-0.8B, _WORKER_SYSTEM only (no code-review prompt)
  Ollama side: qwen2.5:0.5b raw
"""

import subprocess
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

Q3_MODEL_ID = "Qwen/Qwen3.5-0.8B"

_WORKER_SYSTEM = (
    "You are a helpful assistant backed by a structured local memory system. "
    "Use the memory context provided (if any) to give accurate, relevant answers. "
    "Be concise and direct."
)

CALC_FILE  = "test_calculator.py"
TTT_FILE   = "test_tictactoe.js"

SEP = "─" * 64

def load_qwen():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {Q3_MODEL_ID} on {device} ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(Q3_MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        Q3_MODEL_ID, dtype=torch.float16
    ).to(device)
    mdl.eval()
    print(f"Loaded in {(time.perf_counter()-t0)*1000:.0f} ms\n")
    return tok, mdl, device

def qwen_analyze(tok, mdl, device, code: str, filename: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": _WORKER_SYSTEM},
        {"role": "user",   "content": f"Find all the errors in this code:\n\n{code}"},
    ]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )
    ids = enc["input_ids"].to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(
            ids,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    ms = (time.perf_counter() - t0) * 1000
    response = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return response, ms

def ollama_analyze(code: str, filename: str) -> tuple[str, float]:
    prompt = f"Find all the errors in this code:\n\n{code}"
    t0 = time.perf_counter()
    result = subprocess.run(
        ["ollama", "run", "qwen3.5:0.8b", prompt],
        capture_output=True, text=True, timeout=300,
    )
    ms = (time.perf_counter() - t0) * 1000
    return result.stdout.strip(), ms

def run():
    files = {
        CALC_FILE:  open(CALC_FILE).read(),
        TTT_FILE:   open(TTT_FILE).read(),
    }

    # ── TST side ──────────────────────────────────────────────────────────────
    print(SEP)
    print("TST — Qwen3.5-0.8B | _WORKER_SYSTEM (generic, no code-review prompt)")
    print(SEP)
    tok, mdl, device = load_qwen()

    tst_results = {}
    for fname, code in files.items():
        print(f"\n>>> {fname}")
        response, ms = qwen_analyze(tok, mdl, device, code, fname)
        tst_results[fname] = (response, ms)
        print(f"[{ms:.0f} ms]\n{response}\n")

    # ── Ollama side ────────────────────────────────────────────────────────────
    print(SEP)
    print("Ollama — qwen3.5:0.8b raw")
    print(SEP)

    ollama_results = {}
    for fname, code in files.items():
        print(f"\n>>> {fname}")
        response, ms = ollama_analyze(code, fname)
        ollama_results[fname] = (response, ms)
        print(f"[{ms:.0f} ms]\n{response}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(SEP)
    print("TIMING SUMMARY")
    print(SEP)
    for fname in files:
        t_tst, t_ol = tst_results[fname][1], ollama_results[fname][1]
        print(f"{fname}")
        print(f"  TST   : {t_tst:.0f} ms")
        print(f"  Ollama: {t_ol:.0f} ms")

if __name__ == "__main__":
    run()
