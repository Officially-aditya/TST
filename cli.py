"""
cli.py — TST Memory System CLI

Implements the full pipeline from TST_Memory_System_v3.1.md:
    User → Interpreter → Router → Kernel → Worker SLM → Response

Architecture:
  - FunctionGemma 270M  (float32, MPS)  — Tier-1 router
  - Qwen3.5-0.8B        (float16, MPS)  — Tier-2 router + Worker SLM
  - Rust tst_memory kernel via subprocess STDIO

STDIO kernel protocol:
  WRITE {"op":"insert","key":"...","layer":"STM|LTM|Tree","payload":{...}}
  READ  {"keys":[...],"max_results":N}

Usage:
  source gemma-env/bin/activate
  python cli.py [--no-kernel] [--no-worker]

Note: multi-line input is blocked. Enter one query per line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from router.tools import FG_TOOL_SCHEMAS, TOOL_SCHEMAS, VALID_ROUTES

# ─── Config ───────────────────────────────────────────────────────────────────

FG_MODEL_ID   = "google/functiongemma-270m-it"
Q3_MODEL_ID   = "Qwen/Qwen3.5-0.8B"
KERNEL_CWD    = "./tst_memory"
KERNEL_BIN    = "./target/release/server"
ROUTE_TOKENS  = ["route_to_stm", "route_to_ltm", "route_to_tree", "route_to_cloud"]

# FunctionGemma activation phrase (exact — must not change)
_FG_DEVELOPER_MSG = "You are a model that can do function calling with the following functions"

_Q3_ROUTER_SYSTEM = (
    "You are a memory routing engine. "
    "Given a user query, call exactly one of the provided tools. "
    'Respond ONLY with a valid JSON object in this exact format:\n'
    '{"tool_calls": [{"name": "<tool_name>", "args": {<args>}}]}\n'
    "Do not include any explanation or text outside the JSON."
)

_WORKER_SYSTEM = (
    "You are a helpful assistant backed by a structured local memory system. "
    "Use the memory context provided (if any) to give accurate, relevant answers. "
    "Be concise and direct."
)

_WORKER_CODE_SYSTEM = (
    "You are a code review assistant. "
    "Analyze the provided code carefully and list every bug, error, and issue you find. "
    "For each problem: state the line or construct involved, what is wrong, and how to fix it. "
    "Cover all categories: syntax errors (missing colons, brackets, colons after class/def), "
    "semantic errors (wrong operators, undefined variables, wrong return values, division by zero risk), "
    "and logical errors (infinite recursion, off-by-one, incorrect base cases). "
    "Be specific and exhaustive — do not skip any issue."
)

BANNER = """
╔══════════════════════════════════════════════════════╗
║       TST Memory System CLI  v3.1                    ║
║  User → Interpreter → Router → Kernel → Worker SLM  ║
╚══════════════════════════════════════════════════════╝
Type your query or /help for commands.
"""

HELP_TEXT = """
Commands:
  /help               — Show this message
  /status             — Session routing statistics
  /noworker           — Toggle Worker SLM generation (faster route-only testing)
  /analyze <path>     — Analyze a file or directory for bugs (builds Tree DAG)
  /worker             — Show current Worker SLM
  /worker <model-id>  — Swap Worker SLM to any HuggingFace causal-LM
                        Examples:
                          /worker Qwen/Qwen3.5-0.8B
                          /worker microsoft/phi-2
                          /worker mistralai/Mistral-7B-Instruct-v0.2
  /quit               — Exit

Any other input runs the full pipeline:
  1. Interpreter  — extracts query + payload
  2. Router       — classifies tier (STM / LTM / Tree / Cloud)
  3. Kernel       — reads / writes Rust memory
  4. Worker SLM   — generates contextual response
"""

# ─── Global model container ───────────────────────────────────────────────────

@dataclass
class Models:
    device:     str    = "cpu"
    # Tier-1 router (never swapped)
    fg_proc:    object = None
    fg_mdl:     object = None
    # Tier-2 router fallback (never swapped)
    q3_tok:     object = None
    q3_mdl:     object = None
    # Worker SLM (swappable via /worker)
    worker_id:  str    = ""
    worker_tok: object = None
    worker_mdl: object = None

_M = Models()

# ─── Session statistics ───────────────────────────────────────────────────────

@dataclass
class SessionStats:
    total_queries: int   = 0
    route_counts:  dict  = field(default_factory=lambda: {r: 0 for r in ROUTE_TOKENS})
    total_route_ms: float = 0.0
    tier1_hits:    int   = 0
    tier2_hits:    int   = 0
    default_hits:  int   = 0

_STATS = SessionStats()

# ─── Kernel subprocess ────────────────────────────────────────────────────────

class KernelProcess:
    """Manages the Rust tst_memory kernel as a subprocess via STDIO."""

    def __init__(self, cwd: str = KERNEL_CWD):
        self._cwd  = cwd
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        print("  [Kernel] Building tst_memory release binary...")
        build = subprocess.run(
            ["cargo", "build", "--release", "--bin", "server"],
            cwd=self._cwd, capture_output=True, text=True,
        )
        if build.returncode != 0:
            print(f"  [Kernel] Build FAILED:\n{build.stderr[-800:]}")
            return False

        print("  [Kernel] Starting kernel subprocess...")
        self._proc = subprocess.Popen(
            [KERNEL_BIN],
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        ready = self._proc.stdout.readline()
        if "READY" not in ready:
            print(f"  [Kernel] Did not receive READY: {ready!r}")
            return False
        print("  [Kernel] Ready.")
        return True

    def write(self, key: str, layer: str, payload_type: int, data: dict) -> dict:
        if self._proc is None:
            return {"error": "kernel not running"}
        ts = int(time.time() * 1000)
        body = {
            "op":  "insert",
            "key": key,
            "layer": layer,
            "payload": {
                "header": {
                    "payload_type": payload_type,
                    "version":      1,
                    "created_ts":   ts,
                    "last_access_ts": ts,
                    "access_count": 1,
                },
                "data": data,
            },
        }
        self._proc.stdin.write(f"WRITE {json.dumps(body)}\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        try:
            return json.loads(raw)
        except Exception:
            return {"raw": raw.strip()}

    def read(self, keys: list[str]) -> dict:
        if self._proc is None:
            return {"slices": [None] * len(keys)}
        body = {"keys": keys, "max_results": len(keys)}
        self._proc.stdin.write(f"READ {json.dumps(body)}\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        try:
            return json.loads(raw)
        except Exception:
            return {"slices": [None] * len(keys)}

    def tree_insert(self, node_type: str, name: str, parent_id: Optional[int] = None) -> int:
        if self._proc is None:
            return 0
        body = {"node_type": node_type, "name": name, "parent_id": parent_id}
        self._proc.stdin.write(f"TREE_INSERT {json.dumps(body)}\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        try:
            return json.loads(raw).get("node_id", 0)
        except Exception:
            return 0

    def tree_query(self, node_id: int, depth: int = 3) -> list:
        if self._proc is None:
            return []
        body = {"node_id": node_id, "depth": depth}
        self._proc.stdin.write(f"TREE_QUERY {json.dumps(body)}\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        try:
            return json.loads(raw).get("nodes", [])
        except Exception:
            return []

    def tree_link(self, source_id: int, target_id: int, add: bool = True) -> dict:
        if self._proc is None:
            return {}
        body = {"source_id": source_id, "target_id": target_id, "add": add}
        self._proc.stdin.write(f"TREE_LINK {json.dumps(body)}\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def tree_clear(self) -> dict:
        if self._proc is None:
            return {}
        self._proc.stdin.write("TREE_CLEAR\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _key_from(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def _get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"

def _sep(label: str = ""):
    print(f"\n{'─'*56}")
    if label:
        print(f"  {label}")
        print(f"{'─'*56}")

# ─── File parser (lightweight regex, no AST) ──────────────────────────────────

_SKIP_DIRS = {"node_modules", "target", "__pycache__", "venv", "gemma-env", ".git", "dist", "build"}
_SUPPORTED_EXT = {".py", ".js", ".ts", ".jsx", ".tsx", ".rs"}


def _parse_python(src: str) -> dict:
    imports   = re.findall(r'^(?:import|from)\s+([\w.]+)', src, re.MULTILINE)
    functions = re.findall(r'^(?:async\s+)?def\s+(\w+)\s*\(', src, re.MULTILINE)
    classes   = re.findall(r'^class\s+(\w+)', src, re.MULTILINE)
    return {"imports": imports, "functions": functions, "classes": classes}


def _parse_js_ts(src: str) -> dict:
    raw_imp   = re.findall(
        r'(?:import\s+.*?from\s+[\'"]([^\'"\s]+)[\'"]|require\s*\(\s*[\'"]([^\'"\s]+)[\'"]\s*\))',
        src,
    )
    imports   = [a or b for a, b in raw_imp]
    raw_fn    = re.findall(
        r'(?:async\s+)?function\s+(\w+)\s*\(|'
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(',
        src,
    )
    functions = [a or b for a, b in raw_fn]
    classes   = re.findall(r'class\s+(\w+)', src)
    return {"imports": imports, "functions": functions, "classes": classes}


def _parse_rust(src: str) -> dict:
    imports   = re.findall(r'^use\s+([\w:]+)', src, re.MULTILINE)
    mods      = re.findall(r'^(?:pub\s+)?mod\s+(\w+)', src, re.MULTILINE)
    # fn can be indented inside impl blocks — no ^ anchor
    functions = re.findall(r'(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]', src)
    structs   = re.findall(r'^(?:pub\s+)?struct\s+(\w+)', src, re.MULTILINE)
    enums     = re.findall(r'^(?:pub\s+)?enum\s+(\w+)', src, re.MULTILINE)
    return {"imports": imports + mods, "functions": functions, "classes": structs + enums}


def parse_file(path: str) -> dict:
    """Return {imports, functions, classes} extracted from a source file."""
    ext = Path(path).suffix.lower()
    try:
        src = Path(path).read_text(errors="replace")
    except Exception:
        return {"imports": [], "functions": [], "classes": []}
    if ext == ".py":
        return _parse_python(src)
    if ext in (".js", ".ts", ".jsx", ".tsx"):
        return _parse_js_ts(src)
    if ext == ".rs":
        return _parse_rust(src)
    # Generic fallback
    functions = list(dict.fromkeys(re.findall(r'\b(\w+)\s*\(', src)))[:20]
    return {"imports": [], "functions": functions, "classes": []}


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_models():
    _M.device = _get_device()
    print(f"  Device : {_M.device}")

    _sep("Loading FunctionGemma-270M  (float32 — Tier-1 Router)")
    t0 = time.perf_counter()
    _M.fg_proc = AutoProcessor.from_pretrained(FG_MODEL_ID)
    _M.fg_mdl  = AutoModelForCausalLM.from_pretrained(
        FG_MODEL_ID, dtype=torch.float32
    ).to(_M.device)
    _M.fg_mdl.eval()
    print(f"  Loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    _sep("Loading Qwen3.5-0.8B  (float16 — Tier-2 Router + Worker SLM)")
    t0 = time.perf_counter()
    _M.q3_tok = AutoTokenizer.from_pretrained(Q3_MODEL_ID)
    _M.q3_mdl = AutoModelForCausalLM.from_pretrained(
        Q3_MODEL_ID, dtype=torch.float16
    ).to(_M.device)
    _M.q3_mdl.eval()
    print(f"  Loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    # Worker starts as the same model (shared reference, no extra memory)
    _M.worker_id  = Q3_MODEL_ID
    _M.worker_tok = _M.q3_tok
    _M.worker_mdl = _M.q3_mdl

# ─── Worker swap ──────────────────────────────────────────────────────────────

def load_worker(model_id: str) -> bool:
    """
    Hot-swap the Worker SLM to any HuggingFace causal-LM.

    - Frees the old worker from memory unless it is the shared Qwen3.5-0.8B.
    - On failure, reverts to the default worker and returns False.
    - Passing the current model_id is a no-op (prints status, returns True).
    """
    model_id = model_id.strip()

    if model_id == _M.worker_id:
        print(f"  Worker already set to {_M.worker_id!r} — no change.")
        return True

    # Free previous worker only if it was a separately loaded model
    if _M.worker_id != Q3_MODEL_ID and _M.worker_mdl is not None:
        del _M.worker_mdl
        del _M.worker_tok
        _M.worker_mdl = None
        _M.worker_tok = None
        if _M.device == "mps":
            torch.mps.empty_cache()
        elif _M.device == "cuda":
            torch.cuda.empty_cache()

    # Switching back to the default q3 (already in memory)
    if model_id == Q3_MODEL_ID:
        _M.worker_id  = Q3_MODEL_ID
        _M.worker_tok = _M.q3_tok
        _M.worker_mdl = _M.q3_mdl
        print(f"  Worker SLM: restored default {Q3_MODEL_ID!r} (already loaded, no cost).")
        return True

    print(f"  Loading worker: {model_id!r} ...")
    t0 = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16
        ).to(_M.device)
        mdl.eval()
        _M.worker_tok = tok
        _M.worker_mdl = mdl
        _M.worker_id  = model_id
        print(f"  Worker SLM: {model_id!r} ready in {(time.perf_counter()-t0)*1000:.0f} ms")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to load {model_id!r}: {e}")
        # Revert to q3 so the CLI stays usable
        _M.worker_id  = Q3_MODEL_ID
        _M.worker_tok = _M.q3_tok
        _M.worker_mdl = _M.q3_mdl
        print(f"  Reverted to default worker: {Q3_MODEL_ID!r}")
        return False


# ─── Tier-1 Router — FunctionGemma ───────────────────────────────────────────

def _build_fg_prompt(query: str, payload: str):
    user_content = f"Query: '{query}'"
    if payload and payload != query:
        user_content += f"\nPayload: '{payload}'"
    messages = [
        {"role": "developer", "content": _FG_DEVELOPER_MSG},
        {"role": "user",      "content": user_content},
    ]
    return _M.fg_proc.apply_chat_template(
        messages, tools=FG_TOOL_SCHEMAS,
        add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )

def _parse_fg(raw: str) -> Optional[str]:
    m = re.search(r'call:(\w+)\{', raw)
    if m and m.group(1) in VALID_ROUTES:
        return m.group(1)
    for r in ROUTE_TOKENS:
        if r in raw:
            return r
    return None

def _infer_fg(query: str, payload: str) -> tuple[str, float]:
    enc   = _build_fg_prompt(query, payload)
    enc_d = {k: v.to(_M.device) for k, v in enc.items()}
    n_in  = enc_d["input_ids"].shape[1]
    t0    = time.perf_counter()
    with torch.no_grad():
        out = _M.fg_mdl.generate(
            **enc_d,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=_M.fg_proc.eos_token_id,
        )
    ms  = (time.perf_counter() - t0) * 1000.0
    raw = _M.fg_proc.decode(out[0][n_in:], skip_special_tokens=False)
    return raw, ms

# ─── Tier-2 Router — Qwen3.5 fallback ────────────────────────────────────────

def _build_q3_prompt(query: str, payload: str):
    tools_json = json.dumps(TOOL_SCHEMAS, indent=2)
    system = f"{_Q3_ROUTER_SYSTEM}\n\nAvailable tools:\n{tools_json}"
    user   = f"Query: '{query}'"
    if payload and payload != query:
        user += f"\nPayload: '{payload}'"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return _M.q3_tok.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )

def _parse_q3(raw: str) -> Optional[str]:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        calls = obj.get("tool_calls", [])
        if calls and calls[0].get("name") in VALID_ROUTES:
            return calls[0]["name"]
    except Exception:
        pass
    m = re.search(r'\{[\s\S]*?"tool_calls"[\s\S]*?\}', raw)
    if m:
        try:
            obj   = json.loads(m.group())
            calls = obj.get("tool_calls", [])
            if calls and calls[0].get("name") in VALID_ROUTES:
                return calls[0]["name"]
        except Exception:
            pass
    for r in ROUTE_TOKENS:
        if r in raw:
            return r
    return None

def _infer_q3(query: str, payload: str) -> tuple[str, float]:
    enc = _build_q3_prompt(query, payload)
    ids = enc["input_ids"].to(_M.device)
    t0  = time.perf_counter()
    with torch.no_grad():
        out = _M.q3_mdl.generate(
            ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=_M.q3_tok.eos_token_id,
        )
    ms  = (time.perf_counter() - t0) * 1000.0
    raw = _M.q3_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return raw, ms

# ─── Tiered Router ────────────────────────────────────────────────────────────

def route_query(query: str, payload: str) -> tuple[str, float, str]:
    """Returns (route_name, total_ms, tier_label)."""
    raw_fg, ms_fg = _infer_fg(query, payload)
    route = _parse_fg(raw_fg)
    if route:
        _STATS.tier1_hits += 1
        return route, ms_fg, "FunctionGemma-270M"

    raw_q3, ms_q3 = _infer_q3(query, payload)
    route = _parse_q3(raw_q3)
    if route:
        _STATS.tier2_hits += 1
        return route, ms_fg + ms_q3, "Qwen3.5-0.8B (fallback)"

    _STATS.default_hits += 1
    return "route_to_cloud", ms_fg + ms_q3, "default-fallback"

# ─── Interpreter ──────────────────────────────────────────────────────────────

def interpret(user_input: str) -> tuple[str, str, str]:
    """
    Extract (query, payload, file_path) from raw user input.

    query     — the routing classification signal (usually full input)
    payload   — content to store (full input or extracted preference value)
    file_path — file/module path for Tree routes (extracted via regex)
    """
    text      = user_input.strip()
    query     = text
    payload   = text
    file_path = ""

    # Detect file paths mentioned in the input
    fp_m = re.search(
        r'(?:^|\s)([\w./\\-]+\.(?:rs|py|js|ts|go|cpp|c|h|java|rb|sh))\b',
        text,
    )
    if fp_m:
        file_path = fp_m.group(1).strip()

    # Extract preference value (e.g. "I always prefer TypeScript over JavaScript")
    pref_m = re.search(
        r'(?:prefer|always use|should use|use)\s+(\w[\w\s.+-]*?)(?:\s+over|\s+instead|\s*$)',
        text, re.IGNORECASE,
    )
    if pref_m:
        payload = pref_m.group(1).strip()

    return query, payload, file_path

# ─── Kernel Memory Operations ─────────────────────────────────────────────────

def _extract_context(resp: dict) -> str:
    """Pull readable text from a kernel READ response."""
    parts = []
    for s in resp.get("slices", []):
        if s is None:
            continue
        data = s.get("data", {})
        if "TokenStats" in data:
            cf = data["TokenStats"].get("canonical_form", "")
            if cf:
                parts.append(cf)
        elif "Preference" in data:
            p = data["Preference"]
            parts.append(f"{p.get('key','pref')}: {p.get('value','')}")
    return " | ".join(parts)

def kernel_op(
    kernel: KernelProcess,
    route: str,
    query: str,
    payload: str,
    file_path: str,
) -> str:
    """Execute the memory op for the chosen route. Returns context string."""
    key = _key_from(query)

    if route == "route_to_stm":
        kernel.write(key, "STM", 1, {
            "TokenStats": {
                "canonical_form": payload,
                "frequency":      1,
                "decay_score":    1.0,
                "preferred_tokenizer_origin": "cli",
            }
        })
        return _extract_context(kernel.read([key]))

    elif route == "route_to_ltm":
        # Use Preference schema for explicit preference statements
        pref_m = re.search(
            r'(?:prefer|always use|should use)\s+(\w[\w\s.+-]*?)(?:\s+over|\s+instead|\s*$)',
            query, re.IGNORECASE,
        )
        if pref_m:
            data = {
                "Preference": {
                    "key":   "user_preference",
                    "value": payload,
                    "weight": 1.0,
                }
            }
        else:
            data = {
                "TokenStats": {
                    "canonical_form": payload,
                    "frequency":      1,
                    "decay_score":    1.0,
                    "preferred_tokenizer_origin": "cli",
                }
            }
        kernel.write(key, "LTM", 1, data)
        return _extract_context(kernel.read([key]))

    elif route == "route_to_tree":
        # If the tree has been populated (via /analyze), return subgraph context.
        # Node IDs start at 1 after every tree_clear(), so root is always 1.
        tree_nodes = kernel.tree_query(1, 3)
        if tree_nodes:
            parts = []
            for node in tree_nodes[:16]:
                ntype = node.get("node_type", "")
                name  = node.get("name", "")
                parts.append(f"{ntype}:{name}")
            return "Tree: " + ", ".join(parts)
        # Fallback: write a note to the Tree layer
        fp = file_path or _key_from(query)
        kernel.write(key, "Tree", 1, {
            "TokenStats": {
                "canonical_form": f"file:{fp} | {payload}",
                "frequency":      1,
                "decay_score":    1.0,
                "preferred_tokenizer_origin": "cli",
            }
        })
        return _extract_context(kernel.read([key]))

    elif route == "route_to_cloud":
        return ""   # No local kernel interaction

    return ""

# ─── Worker SLM ───────────────────────────────────────────────────────────────

def _has_code(text: str) -> bool:
    """Return True only if the input contains an actual code artifact."""
    return bool(re.search(
        r'```|'                          # fenced code block
        r'def |class |import |from .+ import |'  # Python keywords
        r'function |const |let |var |=>'  # JS/TS
        r'fn |pub |impl |use |mod |'     # Rust
        r'#include|int main|void |'      # C/C++
        r'<\w[\w.-]*>|'                  # HTML/XML tags
        r'\w+\.\w{1,5}:\d+|'            # file:line reference e.g. main.rs:42
        r'(?:^|\s)[\w./\\-]+\.(?:rs|py|js|ts|go|cpp|c|h|java|rb|sh)\b',  # file path
        text,
        re.MULTILINE,
    ))


def worker_respond(user_input: str, route: str, context: str) -> str:
    """Generate a response with Qwen3.5-0.8B, injecting memory context."""
    if route == "route_to_tree" and _has_code(user_input):
        if context:
            system = _WORKER_CODE_SYSTEM + f"\n\nProject structure:\n{context}"
        else:
            system = _WORKER_CODE_SYSTEM
    elif route == "route_to_tree":
        system = _WORKER_SYSTEM   # routed to Tree but no code present — skip review prompt
    elif route == "route_to_cloud":
        system = (
            _WORKER_SYSTEM
            + "\n(No local memory for this query — answering from general knowledge.)"
        )
    elif context:
        system = _WORKER_SYSTEM + f"\n\nMemory context:\n{context}"
    else:
        system = _WORKER_SYSTEM

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_input},
    ]
    tok = _M.worker_tok
    mdl = _M.worker_mdl
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True,
    )
    ids = enc["input_ids"].to(_M.device)
    with torch.no_grad():
        out = mdl.generate(
            ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

# ─── Multi-file Tree analysis ────────────────────────────────────────────────

def _analyze_path(path: str, kernel: Optional[KernelProcess], generate: bool) -> None:
    """
    Analyze a file or directory for bugs.
    Builds the Tree DAG in the Rust kernel, then runs the Worker SLM on each file.
    """
    p = Path(path).expanduser()

    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(
            f for f in p.rglob("*")
            if f.is_file()
            and f.suffix in _SUPPORTED_EXT
            and not any(part in _SKIP_DIRS or part.startswith(".") for part in f.parts)
        )
        if not files:
            print(f"  No supported source files found in {path!r}")
            return
    else:
        print(f"  [ERROR] Not a valid file or directory: {path!r}")
        return

    print(f"  Found {len(files)} file(s) — building Tree DAG...")

    # ── Build Tree in kernel ──────────────────────────────────────────────────
    if kernel:
        kernel.tree_clear()
        proj_id = kernel.tree_insert("Project", p.name)
    else:
        proj_id = None

    file_nodes: dict[str, int] = {}   # rel path → tree node_id
    file_data:  dict[str, dict] = {}  # rel path → {imports, functions, classes}

    for f in files:
        rel = str(f.relative_to(p)) if p.is_dir() else f.name
        parsed = parse_file(str(f))
        file_data[rel] = parsed

        if kernel:
            fid = kernel.tree_insert("File", rel, proj_id)
            file_nodes[rel] = fid
            for fn  in parsed["functions"]:
                kernel.tree_insert("Function", fn, fid)
            for cls in parsed["classes"]:
                kernel.tree_insert("Class", cls, fid)

    # Wire import-based dependency edges between files in the same project
    if kernel and len(files) > 1:
        name_map = {Path(rel).stem: nid for rel, nid in file_nodes.items()}
        for rel, parsed in file_data.items():
            src_id = file_nodes.get(rel)
            if not src_id:
                continue
            for imp in parsed["imports"]:
                stem = imp.split(".")[-1].split("/")[-1]
                tgt_id = name_map.get(stem)
                if tgt_id and tgt_id != src_id:
                    kernel.tree_link(src_id, tgt_id)

    # ── Print tree summary ────────────────────────────────────────────────────
    print("\n  Tree structure:")
    for rel, parsed in file_data.items():
        print(f"    {rel}")
        if parsed["functions"]:
            fn_str = ", ".join(parsed["functions"][:5])
            suffix = "…" if len(parsed["functions"]) > 5 else ""
            print(f"      fn : {fn_str}{suffix}")
        if parsed["classes"]:
            cls_str = ", ".join(parsed["classes"][:5])
            suffix  = "…" if len(parsed["classes"]) > 5 else ""
            print(f"      cls: {cls_str}{suffix}")
        if parsed["imports"]:
            imp_str = ", ".join(parsed["imports"][:4])
            suffix  = "…" if len(parsed["imports"]) > 4 else ""
            print(f"      imp: {imp_str}{suffix}")

    if not generate:
        print("\n  [Worker OFF] — route-only mode, skipping code review.")
        return

    # ── Code review per file ──────────────────────────────────────────────────
    SEP = "─" * 56
    print(f"\n{SEP}")
    print("  Code Review")
    print(SEP)

    for f in files:
        rel = str(f.relative_to(p)) if p.is_dir() else f.name
        try:
            code = f.read_text(errors="replace")
        except Exception as e:
            print(f"\n  [ERROR] Cannot read {rel}: {e}")
            continue

        n_lines = len(code.splitlines())
        print(f"\n  >>> {rel}  ({n_lines} lines)")

        # Build project-structure context from parsed data
        parsed = file_data[rel]
        ctx_parts = []
        if parsed["functions"]:
            ctx_parts.append(f"Functions: {', '.join(parsed['functions'])}")
        if parsed["classes"]:
            ctx_parts.append(f"Classes: {', '.join(parsed['classes'])}")
        if parsed["imports"]:
            ctx_parts.append(f"Imports: {', '.join(parsed['imports'][:6])}")
        if len(files) > 1:
            others = [r for r in file_data if r != rel]
            if others:
                ctx_parts.append(f"Other files: {', '.join(others[:4])}")
        context = " | ".join(ctx_parts)

        user_msg = f"Find all bugs and errors in this code:\n\n{code}"
        t0 = time.perf_counter()
        response = worker_respond(user_msg, "route_to_tree", context)
        ms = (time.perf_counter() - t0) * 1000

        print(f"  [{ms:.0f} ms]\n")
        print(f"  {response}\n")

    print(SEP)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    user_input: str,
    kernel: Optional[KernelProcess],
    generate_response: bool = True,
) -> dict:
    """
    User → Interpreter → Router → Kernel → Worker SLM

    Returns a result dict with route, tier, context, response, latencies.
    """
    t_start = time.perf_counter()

    # 1. Interpret
    query, payload, file_path = interpret(user_input)

    # 2. Route (tiered: FG → Qwen → default)
    route, route_ms, tier = route_query(query, payload)
    _STATS.total_queries += 1
    _STATS.route_counts[route] = _STATS.route_counts.get(route, 0) + 1
    _STATS.total_route_ms     += route_ms

    # 3. Kernel memory op
    context = ""
    if kernel:
        context = kernel_op(kernel, route, query, payload, file_path)

    # 4. Worker SLM response
    response  = ""
    worker_ms = 0.0
    if generate_response:
        t_w      = time.perf_counter()
        response = worker_respond(user_input, route, context)
        worker_ms = (time.perf_counter() - t_w) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "route":     route,
        "tier":      tier,
        "context":   context,
        "response":  response,
        "route_ms":  round(route_ms,  1),
        "worker_ms": round(worker_ms, 1),
        "total_ms":  round(total_ms,  1),
    }

# ─── Status display ───────────────────────────────────────────────────────────

def print_status():
    n = _STATS.total_queries
    print(f"\n  Session stats  ({n} queries processed):")
    print(f"  Worker SLM          : {_M.worker_id}")
    if n == 0:
        print("  No queries yet.")
        return
    avg_r = _STATS.total_route_ms / n
    print(f"  Avg router latency  : {avg_r:.0f} ms")
    print(f"  Tier-1 hits (FG)    : {_STATS.tier1_hits}/{n}")
    print(f"  Tier-2 hits (Qwen)  : {_STATS.tier2_hits}/{n}")
    print(f"  Default fallback    : {_STATS.default_hits}/{n}")
    print("  Route distribution  :")
    for route, count in _STATS.route_counts.items():
        bar = "█" * count
        short = route.replace("route_to_", "")
        print(f"    {short:<8} {count:>3}  {bar}")

# ─── Input reader (single-line or multi-line block) ───────────────────────────

def read_input() -> str:
    """
    Read one line of input.

    Detects multi-line pastes by checking if stdin has more data buffered
    immediately after input() returns (only happens when text is pasted).
    Drains and blocks the paste rather than firing the pipeline on every line.
    """
    line = input("\n[TST] > ").strip()

    # select with timeout=0: returns immediately if stdin has buffered data.
    # A single typed line leaves stdin empty; a paste leaves all remaining
    # lines sitting in the buffer.
    if select.select([sys.stdin], [], [], 0)[0]:
        # Drain every buffered line so the REPL loop stays clean.
        drained = 0
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            drained += 1
        print(f"  [BLOCKED] Multi-line paste detected ({drained + 1} lines). "
              "Enter one line at a time.")
        return ""

    return line

# ─── REPL ─────────────────────────────────────────────────────────────────────

def _display_result(result: dict, generate: bool):
    route_short = result["route"].replace("route_to_", "").upper()
    print(
        f"\n  [{route_short} | {result['tier']} | "
        f"router {result['route_ms']:.0f} ms]"
    )
    if result["context"]:
        print(f"  [Memory] {result['context'][:120]}")
    if result["response"]:
        print(f"\n  {result['response']}\n")
    elif generate and result["route"] == "route_to_cloud":
        print("  [Cloud] This query needs an external LLM — no local memory available.\n")
    elif generate:
        print("  [Worker] No response generated.\n")


def repl(kernel: Optional[KernelProcess]):
    generate = True
    print(BANNER)

    try:
        import readline  # noqa: F401 — arrow-key history on macOS/Linux
    except ImportError:
        pass

    while True:
        try:
            user_input = read_input()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd_lower = user_input.lower().strip()

            if cmd_lower in ("/quit", "/exit", "/q"):
                print("Goodbye.")
                break

            elif cmd_lower == "/help":
                print(HELP_TEXT)
                continue

            elif cmd_lower == "/status":
                print_status()
                continue

            elif cmd_lower == "/noworker":
                generate = not generate
                state = "ON" if generate else "OFF (route-only)"
                print(f"  Worker SLM: {state}")
                continue

            elif cmd_lower.startswith("/worker"):
                parts = user_input.split(None, 1)
                if len(parts) < 2 or not parts[1].strip():
                    print(f"  Current worker: {_M.worker_id}")
                    print("  Usage: /worker <huggingface-model-id>")
                else:
                    load_worker(parts[1].strip())
                continue

            elif cmd_lower.startswith("/analyze"):
                parts = user_input.split(None, 1)
                if len(parts) < 2 or not parts[1].strip():
                    print("  Usage: /analyze <file_or_directory>")
                    continue
                _analyze_path(parts[1].strip(), kernel, generate)
                continue

            else:
                print(f"  Unknown command '{user_input}'. Type /help.")
                continue

        # ── Pipeline ──────────────────────────────────────────────────────────
        try:
            result = run_pipeline(user_input, kernel, generate_response=generate)
        except Exception as e:
            print(f"  [ERROR] Pipeline failed: {e}")
            import traceback; traceback.print_exc()
            continue

        _display_result(result, generate)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TST Memory System CLI — full pipeline REPL",
    )
    parser.add_argument(
        "--no-kernel", action="store_true",
        help="Skip starting the Rust kernel (useful for router-only testing)",
    )
    parser.add_argument(
        "--no-worker", action="store_true",
        help="Skip Worker SLM generation (route-only mode, much faster)",
    )
    args = parser.parse_args()

    _sep("TST Memory System CLI v3.1 — Startup")

    # 1. Start Rust kernel
    kernel: Optional[KernelProcess] = None
    if not args.no_kernel:
        kernel = KernelProcess(cwd=KERNEL_CWD)
        ok = kernel.start()
        if not ok:
            print("  [WARNING] Kernel failed — running without memory persistence.")
            kernel = None
    else:
        print("  [INFO] --no-kernel flag set: skipping Rust kernel.")

    # 2. Load models
    load_models()
    print("\n  All components ready.")

    # 3. REPL
    try:
        repl(kernel)
    finally:
        if kernel:
            kernel.stop()
            print("  [Kernel] Stopped.")

if __name__ == "__main__":
    main()
