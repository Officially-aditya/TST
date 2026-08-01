"""Thin FastAPI wrapper around the shared action router and STDIO kernel client.

The Rust kernel is a local subprocess. There is deliberately no Rust HTTP URL;
the CLI, this server, and evaluation scripts all use ``StdioKernelClient``.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from router.tools import ActionHandlers
from tst.kernel.client import StdioKernelClient
from tst.kernel.process import KernelProcessConfig
from tst.routing.decision import RouteDecision
from tst.routing.functiongemma import DEVELOPER_MESSAGE
from tst.routing.router import ActionRouter
from tst.routing.schemas import FG_TOOL_SCHEMAS, TOOL_SCHEMAS

log = logging.getLogger("tst.router")

FG_MODEL_ID = os.environ.get("TST_ROUTER_TIER1_MODEL", "google/functiongemma-270m-it")
QWEN_MODEL_ID = os.environ.get("TST_ROUTER_TIER2_MODEL", "Qwen/Qwen3.5-0.8B")
QWEN_SYSTEM = (
    "You are a memory action router. Call exactly one provided tool. "
    'Return only {"tool_calls":[{"name":"...","args":{...}}]} with no prose.'
)


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=100_000)
    payload: str = Field(default="", max_length=1_000_000)


class Telemetry(BaseModel):
    source: str
    model: str | None = None
    wall_time_ms: float
    kernel_ms: float | None = None
    parse_failures: int = 0
    inference_failures: int = 0
    tier_usage: dict[str, int] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    query: str
    decision: RouteDecision
    tool_called: str
    args: dict[str, Any]
    result: dict[str, Any]
    telemetry: Telemetry


@dataclass(slots=True)
class _ModelState:
    device: str = "cpu"
    torch: Any = None
    fg_processor: Any = None
    fg_model: Any = None
    qwen_tokenizer: Any = None
    qwen_model: Any = None

    @property
    def loaded(self) -> bool:
        return self.fg_model is not None and self.qwen_model is not None


_MODELS = _ModelState()
_KERNEL: StdioKernelClient | None = None
_ROUTER: ActionRouter | None = None
_ROUTE_LOCK = threading.Lock()
_OPERATION_LAYER_COUNTS: Counter[str] = Counter()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _external_provider_configured() -> bool:
    return bool(os.environ.get("TST_EXTERNAL_PROVIDER", "").strip())


def _load_models() -> None:
    # Imports stay local so protocol/routing users do not need model packages.
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fg_processor = AutoProcessor.from_pretrained(FG_MODEL_ID)
    fg_model: Any = AutoModelForCausalLM.from_pretrained(FG_MODEL_ID, dtype=torch.float32)
    fg_model = fg_model.to(device)
    fg_model.eval()

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen_dtype = torch.float16 if device != "cpu" else torch.float32
    qwen_model: Any = AutoModelForImageTextToText.from_pretrained(QWEN_MODEL_ID, dtype=qwen_dtype)
    qwen_model = qwen_model.to(device)
    qwen_model.eval()

    _MODELS.torch = torch
    _MODELS.device = device
    _MODELS.fg_processor = fg_processor
    _MODELS.fg_model = fg_model
    _MODELS.qwen_tokenizer = qwen_tokenizer
    _MODELS.qwen_model = qwen_model


def _unload_models() -> None:
    torch = _MODELS.torch
    _MODELS.fg_processor = None
    _MODELS.fg_model = None
    _MODELS.qwen_tokenizer = None
    _MODELS.qwen_model = None
    _MODELS.torch = None
    _MODELS.device = "cpu"
    gc.collect()
    if torch is not None:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _infer_functiongemma(query: str, payload: str) -> str:
    processor = _MODELS.fg_processor
    model = _MODELS.fg_model
    if processor is None or model is None:
        raise RuntimeError("FunctionGemma is not loaded")
    content = f"Query: {query!r}"
    if payload:
        content += f"\nPayload: {payload!r}"
    encoded = processor.apply_chat_template(
        [
            {"role": "developer", "content": DEVELOPER_MESSAGE},
            {"role": "user", "content": content},
        ],
        tools=FG_TOOL_SCHEMAS,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {name: value.to(_MODELS.device) for name, value in encoded.items()}
    input_size = inputs["input_ids"].shape[1]
    with _MODELS.torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=processor.eos_token_id,
        )
    return processor.decode(output[0][input_size:], skip_special_tokens=False)


def _infer_qwen(query: str, payload: str) -> str:
    tokenizer = _MODELS.qwen_tokenizer
    model = _MODELS.qwen_model
    if tokenizer is None or model is None:
        raise RuntimeError("Qwen router is not loaded")
    system = f"{QWEN_SYSTEM}\n\nAvailable tools:\n{json.dumps(TOOL_SCHEMAS)}"
    content = f"Query: {query!r}"
    if payload:
        content += f"\nPayload: {payload!r}"
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {name: value.to(_MODELS.device) for name, value in encoded.items()}
    input_size = inputs["input_ids"].shape[1]
    with _MODELS.torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_size:], skip_special_tokens=True)


def run_route(
    query: str,
    payload: str = "",
    *,
    client: StdioKernelClient | None = None,
    action_router: ActionRouter | None = None,
) -> RouteResponse:
    """Route and dispatch one query; dependency injection keeps tests model-free."""

    started = time.perf_counter()
    router = action_router or _ROUTER
    if router is None:
        router = ActionRouter(
            _infer_functiongemma if _MODELS.fg_model is not None else None,
            _infer_qwen if _MODELS.qwen_model is not None else None,
            external_provider_configured=_external_provider_configured(),
        )
    with _ROUTE_LOCK:
        decision = router.route(query, payload)
        _OPERATION_LAYER_COUNTS[f"{decision.operation}:{decision.layer}"] += 1
    active_client = client if client is not None else _KERNEL
    handlers = ActionHandlers(
        active_client,
        external_provider_configured=_external_provider_configured(),
    )
    result = handlers.dispatch(decision, query)
    wall_ms = (time.perf_counter() - started) * 1000.0
    kernel_ms = None
    kernel = result.get("kernel")
    if isinstance(kernel, dict) and isinstance(kernel.get("metrics"), dict):
        kernel_ms = kernel["metrics"].get("kernel_ms")

    return RouteResponse(
        query=query,
        decision=decision,
        tool_called=_tool_for(decision.operation),
        args={
            key: value
            for key, value in decision.model_dump().items()
            if key not in {"operation", "confidence", "source"} and value not in (None, [], "")
        },
        result=result,
        telemetry=Telemetry(
            source=decision.source,
            model={
                "functiongemma": FG_MODEL_ID,
                "qwen": QWEN_MODEL_ID,
            }.get(decision.source),
            wall_time_ms=round(wall_ms, 3),
            kernel_ms=kernel_ms,
            parse_failures=router.parse_failures,
            inference_failures=router.inference_failures,
            tier_usage=dict(router.tier_usage),
        ),
    )


def _tool_for(operation: str) -> str:
    return {
        "store": "store_memory",
        "retrieve": "retrieve_memory",
        "update": "update_memory",
        "forget": "forget_memory",
        "search": "search_memory",
        "analyze_code": "query_code_graph",
        "answer_without_memory": "answer_without_memory",
        "escalate_external": "escalate_external",
    }[operation]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _KERNEL, _ROUTER
    try:
        if not _env_enabled("TST_ROUTER_NO_KERNEL"):
            config = KernelProcessConfig(build_kernel=_env_enabled("TST_BUILD_KERNEL"))
            _KERNEL = StdioKernelClient(config)
            _KERNEL.start()
        if not _env_enabled("TST_ROUTER_SKIP_MODELS"):
            _load_models()
        _OPERATION_LAYER_COUNTS.clear()
        _ROUTER = ActionRouter(
            _infer_functiongemma if _MODELS.fg_model is not None else None,
            _infer_qwen if _MODELS.qwen_model is not None else None,
            external_provider_configured=_external_provider_configured(),
        )
        yield
    finally:
        if _KERNEL is not None:
            _KERNEL.close(graceful=True)
            _KERNEL = None
        _ROUTER = None
        _OPERATION_LAYER_COUNTS.clear()
        _unload_models()


app = FastAPI(
    title="TST Memory Router",
    description="Action-aware local router using the versioned TST STDIO kernel.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    try:
        return run_route(request.query, request.payload)
    except Exception as exc:
        log.exception("routing failed")
        raise HTTPException(status_code=500, detail="routing failed") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    kernel_ok = _env_enabled("TST_ROUTER_NO_KERNEL") or bool(_KERNEL and _KERNEL.is_running)
    models_ok = _env_enabled("TST_ROUTER_SKIP_MODELS") or _MODELS.loaded
    return {
        "status": "ok" if kernel_ok and models_ok else "degraded",
        "kernel_running": bool(_KERNEL and _KERNEL.is_running),
        "models_loaded": _MODELS.loaded,
        "device": _MODELS.device,
    }


@app.get("/status")
def status() -> dict[str, Any]:
    kernel_status: dict[str, Any] | None = None
    kernel_error: str | None = None
    if _KERNEL is not None and _KERNEL.is_running:
        try:
            kernel_status = _KERNEL.status()
        except Exception:
            log.exception("kernel status request failed")
            kernel_error = "kernel status unavailable"
    with _ROUTE_LOCK:
        operation_layer_confusion: dict[str, dict[str, int]] = {}
        for key, count in sorted(_OPERATION_LAYER_COUNTS.items()):
            operation, layer = key.split(":", maxsplit=1)
            operation_layer_confusion.setdefault(operation, {})[layer] = count
        router_status = {
            "tier_usage": dict(_ROUTER.tier_usage) if _ROUTER is not None else {},
            "parse_failures": _ROUTER.parse_failures if _ROUTER is not None else 0,
            "inference_failures": _ROUTER.inference_failures if _ROUTER is not None else 0,
            "operation_layer_confusion": operation_layer_confusion,
        }
    return {
        **health(),
        "kernel": kernel_status,
        "kernel_error": kernel_error,
        "router": router_status,
        "worker": {"validation_failures": 0},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("router.server:app", host="127.0.0.1", port=8003, reload=False)
