"""
router/server.py

FastAPI routing server — TST Memory Router.

Tiered inference:
  Tier 1 — FunctionGemma 270M  (google/functiongemma-270m-it)
            Primary router. Uses native function-calling format:
              - role "developer" (not "system")
              - tools passed via apply_chat_template(tools=[...])
              - output: <start_function_call>call:name{p:<escape>v<escape>}<end_function_call>
            Logit bias applied via TSTBiasProcessor (from Rust bias.rs).
            Target latency: <500 ms (warm).

  Tier 2 — Qwen3.5-0.8B  (Qwen/Qwen3.5-0.8B)
            Fallback when FunctionGemma emits no valid tool_call.
            Uses structured JSON tool-call prompt + JSON output parser.
            Target latency: ~1.4 s (warm).

  Default — route_to_cloud
            If both tiers fail to produce a parseable tool_call.

Endpoints:
  POST /route   — classify query and dispatch to Rust Kernel
  GET  /health  — liveness check

Run:
  python -m router.server          (from v2/)
  or
  uvicorn router.server:app --port 8003
"""

from __future__ import annotations

import json
import re
import time
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,      # FunctionGemma requires AutoProcessor, not AutoTokenizer
    AutoTokenizer,      # Qwen3 still uses AutoTokenizer
    LogitsProcessor,
    LogitsProcessorList,
)

from router.tools import TOOL_SCHEMAS, FG_TOOL_SCHEMAS, TOOL_MAP, VALID_ROUTES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("router")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FG_MODEL_ID  = "google/functiongemma-270m-it"   # Tier 1
Q3_MODEL_ID  = "Qwen/Qwen3.5-0.8B"              # Tier 2 (cached locally)
KERNEL_URL   = "http://127.0.0.1:3000"

# Route token strings as they appear in FunctionGemma output
ROUTE_TOKENS = ["route_to_stm", "route_to_ltm", "route_to_tree", "route_to_cloud"]

# FunctionGemma system message — must use role "developer" and this exact text.
# Tools are injected by apply_chat_template(tools=FG_TOOL_SCHEMAS), NOT in the message body.
_FG_DEVELOPER_MSG = "You are a model that can do function calling with the following functions"

# Qwen3 system prompt — JSON tool-call format embedded in message body.
_Q3_SYSTEM_PROMPT = (
    "You are a memory routing engine. "
    "Given a user query, call exactly one of the provided tools. "
    'Respond ONLY with a valid JSON object in this exact format:\n'
    '{"tool_calls": [{"name": "<tool_name>", "args": {<args>}}]}\n'
    "Do not include any explanation or text outside the JSON."
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    query:   str = Field(description="Natural language query to route")
    payload: str = Field(default="", description="Optional payload content")

class KernelResult(BaseModel):
    route:     str
    key:       str | None
    escalate:  bool = False
    raw_response: dict[str, Any] = {}

class Telemetry(BaseModel):
    tier:            str
    model:           str
    prompt_tokens:   int
    eval_tokens:     int
    total_tokens:    int
    wall_time_ms:    float
    tokens_per_sec:  float

class RouteResponse(BaseModel):
    query:        str
    tool_called:  str
    args:         dict[str, Any]
    result:       KernelResult
    telemetry:    Telemetry

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

class _Models:
    fg_processor:  AutoProcessor           | None = None   # FunctionGemma uses AutoProcessor
    fg_model:      AutoModelForCausalLM    | None = None
    q3_tokenizer:  AutoTokenizer           | None = None   # Qwen uses AutoTokenizer
    q3_model:      AutoModelForCausalLM    | None = None
    device:        str                            = "cpu"
    route_token_ids: dict[str, int]               = {}

_M = _Models()

# ---------------------------------------------------------------------------
# TSTBiasProcessor
# ---------------------------------------------------------------------------

class TSTBiasProcessor(LogitsProcessor):
    """
    Fetches live logit biases from the Rust Kernel (bias.rs) and applies them
    to the model's logit tensor before sampling.

    The Rust Kernel tracks per-route access frequency and decay. Routes with
    warm context receive a positive bias nudging the model toward them —
    without overriding its own judgment when the query clearly belongs
    elsewhere.

    Bias endpoint: GET /bias  → {"route_to_stm": 0.3, "route_to_ltm": -0.1, ...}
    """

    def __init__(self, route_token_ids: dict[str, int]):
        self._ids = route_token_ids
        self._bias_map: dict[str, float] = {}
        self._fetched = False

    def _fetch(self) -> None:
        if self._fetched:
            return
        try:
            r = httpx.get(f"{KERNEL_URL}/bias", timeout=0.5)
            if r.status_code == 200:
                self._bias_map = r.json()
        except Exception:
            pass  # Kernel unavailable — proceed without bias
        self._fetched = True

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores:    torch.FloatTensor,
    ) -> torch.FloatTensor:
        self._fetch()
        for route_name, bias_val in self._bias_map.items():
            token_id = self._ids.get(route_name)
            if token_id is not None:
                scores[:, token_id] += bias_val
        return scores

# ---------------------------------------------------------------------------
# Prompt builders — one per model family
# ---------------------------------------------------------------------------

def _build_fg_prompt(processor: AutoProcessor, query: str, payload: str) -> dict:
    """
    FunctionGemma native format:
      - role "developer" (not "system") with the fixed activation phrase
      - tools injected via apply_chat_template(tools=FG_TOOL_SCHEMAS)
      - tools are NOT embedded in the message body
    Output will be: <start_function_call>call:name{p:<escape>v<escape>}<end_function_call>
    """
    user_content = f"Query: '{query}'"
    if payload:
        user_content += f"\nPayload: '{payload}'"

    messages = [
        {"role": "developer", "content": _FG_DEVELOPER_MSG},
        {"role": "user",      "content": user_content},
    ]
    return processor.apply_chat_template(
        messages,
        tools=FG_TOOL_SCHEMAS,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )


def _build_q3_prompt(tokenizer: AutoTokenizer, query: str, payload: str) -> dict:
    """
    Qwen3 format: structured JSON tool-call prompt embedded in system message.
    Output expected: {"tool_calls": [{"name": "...", "args": {...}}]}
    """
    tools_json = json.dumps(TOOL_SCHEMAS, indent=2)
    system = f"{_Q3_SYSTEM_PROMPT}\n\nAvailable tools:\n{tools_json}"
    user_content = f"Query: '{query}'"
    if payload:
        user_content += f"\nPayload: '{payload}'"

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

# ---------------------------------------------------------------------------
# Output parsers — one per model family
# ---------------------------------------------------------------------------

def _parse_fg_output(raw: str) -> dict | None:
    """
    Parse FunctionGemma native output:
      <start_function_call>call:route_to_stm{query:<escape>...<escape>,payload:<escape>...<escape>}<end_function_call>

    Extracts the function name and best-effort parses args from the <escape>-wrapped values.
    Falls back to substring scan on the function name if the structured parse fails.
    """
    raw = raw.strip()

    # Attempt 1 — structured parse of call:name{...} block
    name_match = re.search(r'call:(\w+)\{', raw)
    if name_match:
        name = name_match.group(1)
        if name in VALID_ROUTES:
            args: dict[str, str] = {}
            # Extract param:<escape>value<escape> pairs
            for m in re.finditer(r'(\w+):<escape>(.*?)<escape>', raw):
                args[m.group(1)] = m.group(2)
            return {"name": name, "args": args}

    # Attempt 2 — substring scan (handles malformed output)
    for route in ROUTE_TOKENS:
        if route in raw:
            return {"name": route, "args": {}}

    return None


def _parse_q3_output(raw: str) -> dict | None:
    """
    Parse Qwen3 JSON output: {"tool_calls": [{"name": "...", "args": {...}}]}

    Attempt order:
      1. Direct json.loads
      2. Regex to extract first {...} block containing "tool_calls"
      3. Substring scan for a known route name (no args)
    """
    raw = raw.strip()

    try:
        obj = json.loads(raw)
        calls = obj.get("tool_calls", [])
        if calls and calls[0].get("name") in VALID_ROUTES:
            return {"name": calls[0]["name"], "args": calls[0].get("args", {})}
    except (json.JSONDecodeError, AttributeError):
        pass

    match = re.search(r'\{[\s\S]*?"tool_calls"[\s\S]*?\}', raw)
    if match:
        try:
            obj = json.loads(match.group())
            calls = obj.get("tool_calls", [])
            if calls and calls[0].get("name") in VALID_ROUTES:
                return {"name": calls[0]["name"], "args": calls[0].get("args", {})}
        except (json.JSONDecodeError, AttributeError):
            pass

    for route in ROUTE_TOKENS:
        if route in raw:
            return {"name": route, "args": {}}

    return None

# ---------------------------------------------------------------------------
# Telemetry builder
# ---------------------------------------------------------------------------

def _build_telemetry(
    tier:    str,
    model:   str,
    n_input: int,
    n_new:   int,
    wall_ms: float,
) -> Telemetry:
    total = n_input + n_new
    tps   = round(n_new / (wall_ms / 1000.0), 1) if wall_ms > 0 else 0.0
    return Telemetry(
        tier=tier,
        model=model,
        prompt_tokens=n_input,
        eval_tokens=n_new,
        total_tokens=total,
        wall_time_ms=round(wall_ms, 1),
        tokens_per_sec=tps,
    )

# ---------------------------------------------------------------------------
# Core inference — one per model family
# ---------------------------------------------------------------------------

def _infer_fg(query: str, payload: str) -> tuple[str, int, int, float]:
    """
    FunctionGemma inference pass.
    Uses AutoProcessor + developer role + FG_TOOL_SCHEMAS via apply_chat_template.
    Returns (raw_text, n_input_tokens, n_new_tokens, wall_ms).
    skip_special_tokens=False so <start_function_call>/<end_function_call> are preserved.
    """
    encoded = _build_fg_prompt(_M.fg_processor, query, payload)
    # Pass ALL encoded tensors (input_ids + attention_mask) — without attention_mask
    # FunctionGemma generates only <pad> tokens and produces no usable output.
    encoded_on_device = {k: v.to(_M.device) for k, v in encoded.items()}
    n_input = encoded_on_device["input_ids"].shape[1]

    lp_list: list[LogitsProcessor] = []
    if _M.route_token_ids:
        lp_list.append(TSTBiasProcessor(_M.route_token_ids))

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = _M.fg_model.generate(
            **encoded_on_device,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=_M.fg_processor.eos_token_id,
            logits_processor=LogitsProcessorList(lp_list) if lp_list else None,
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    new_ids = output_ids[0][n_input:]
    raw = _M.fg_processor.decode(new_ids, skip_special_tokens=False)
    return raw, n_input, new_ids.shape[0], wall_ms


def _infer_q3(query: str, payload: str) -> tuple[str, int, int, float]:
    """
    Qwen3 inference pass.
    Uses AutoTokenizer + system role + JSON-in-prompt format.
    Returns (raw_text, n_input_tokens, n_new_tokens, wall_ms).
    """
    encoded   = _build_q3_prompt(_M.q3_tokenizer, query, payload)
    input_ids = encoded["input_ids"].to(_M.device)
    n_input   = input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = _M.q3_model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=_M.q3_tokenizer.eos_token_id,
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    new_ids = output_ids[0][n_input:]
    raw = _M.q3_tokenizer.decode(new_ids, skip_special_tokens=True)
    return raw, n_input, new_ids.shape[0], wall_ms

# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------

def run_route(query: str, payload: str) -> RouteResponse:
    wall_start = time.perf_counter()

    # ── Tier 1: FunctionGemma ────────────────────────────────────────────
    raw, n_in, n_new, t_ms = _infer_fg(query, payload)
    log.info("FG raw output: %s", raw[:300])
    call = _parse_fg_output(raw)
    tier  = "functiongemma-270m"
    model = FG_MODEL_ID

    # ── Tier 2 fallback: Qwen3 0.8B ──────────────────────────────────────
    if call is None:
        log.info("Tier 1 produced no valid tool_call — escalating to Qwen3.5-0.8B")
        raw, n_in, n_new, t_ms = _infer_q3(query, payload)
        log.info("Q3 raw output: %s", raw[:200])
        call  = _parse_q3_output(raw)
        tier  = "qwen3.5-0.8b"
        model = Q3_MODEL_ID

    # ── Default fallback ─────────────────────────────────────────────────
    if call is None:
        log.warning("Both tiers failed — defaulting to route_to_cloud")
        call  = {"name": "route_to_cloud", "args": {"query": query}}
        tier  = "fallback-default"
        model = "none"
        n_in  = 0
        n_new = 0
        t_ms  = 0.0

    # Ensure query is always in args
    call["args"].setdefault("query", query)
    if call["name"] in ("route_to_stm", "route_to_ltm") and "payload" not in call["args"]:
        call["args"]["payload"] = payload
    if call["name"] == "route_to_tree" and "file_path" not in call["args"]:
        call["args"]["file_path"] = ""

    # ── Dispatch to Rust Kernel ───────────────────────────────────────────
    dispatch_fn = TOOL_MAP[call["name"]]
    kernel_raw  = dispatch_fn(**call["args"])

    result = KernelResult(
        route=kernel_raw.get("route", call["name"]),
        key=kernel_raw.get("key"),
        escalate=kernel_raw.get("escalate", False),
        raw_response=kernel_raw,
    )

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return RouteResponse(
        query=query,
        tool_called=call["name"],
        args=call["args"],
        result=result,
        telemetry=_build_telemetry(tier, model, n_in, n_new, wall_ms),
    )

# ---------------------------------------------------------------------------
# FastAPI lifespan — load models once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models...")
    _M.device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", _M.device)

    # Tier 1 — FunctionGemma 270M  (AutoProcessor, float32)
    # MUST use float32 — RecurrentGemma SSM layers produce NaN/pad tokens under float16.
    log.info("Loading %s (float32) ...", FG_MODEL_ID)
    _M.fg_processor = AutoProcessor.from_pretrained(FG_MODEL_ID)
    _M.fg_model = AutoModelForCausalLM.from_pretrained(
        FG_MODEL_ID, dtype=torch.float32
    ).to(_M.device)
    _M.fg_model.eval()

    # Pre-resolve route token IDs for the bias processor
    _M.route_token_ids = {}
    for route in ROUTE_TOKENS:
        ids = _M.fg_processor.tokenizer.encode(route, add_special_tokens=False)
        if ids:
            _M.route_token_ids[route] = ids[0]
    log.info("Route token IDs: %s", _M.route_token_ids)

    # Tier 2 — Qwen3 0.6B
    log.info("Loading %s ...", Q3_MODEL_ID)
    _M.q3_tokenizer = AutoTokenizer.from_pretrained(Q3_MODEL_ID)
    _M.q3_model = AutoModelForCausalLM.from_pretrained(
        Q3_MODEL_ID, torch_dtype=torch.float16
    ).to(_M.device)
    _M.q3_model.eval()

    log.info("All models ready.")
    yield

    # Cleanup
    log.info("Shutting down.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TST Memory Router",
    description="Tiered LLM router (FunctionGemma → Qwen3 0.6B) dispatching to the Rust TST Kernel.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    """Classify the query and dispatch to the appropriate memory tier."""
    try:
        return run_route(request.query, request.payload)
    except Exception as e:
        log.exception("Routing error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "tier1":  FG_MODEL_ID,
        "tier2":  Q3_MODEL_ID,
        "device": _M.device,
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("router.server:app", host="127.0.0.1", port=8003, reload=False)
