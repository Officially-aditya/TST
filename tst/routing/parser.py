"""Strict model-output parsers. Malformed prose never becomes a mutation."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from pydantic import ValidationError

from .decision import RouteDecision, RouteLayer, RouteOperation, RouteSource

TOOL_NAMES = {
    "store_memory",
    "retrieve_memory",
    "update_memory",
    "forget_memory",
    "search_memory",
    "query_code_graph",
    "answer_without_memory",
    "escalate_external",
}


class RouteParseError(ValueError):
    pass


def parse_functiongemma_output(
    raw: str,
    *,
    query: str,
    source: RouteSource = "functiongemma",
    confidence: float = 0.82,
) -> RouteDecision:
    text = raw.strip()
    if text.startswith("{"):
        return parse_json_tool_output(text, query=query, source=source, confidence=confidence)

    match = re.fullmatch(
        r"(?:<start_function_call>)?call:(?P<name>[a-z_]+)\{(?P<body>.*)\}"
        r"(?:<end_function_call>)?",
        text,
        re.DOTALL,
    )
    if not match:
        raise RouteParseError("not a complete FunctionGemma tool call")
    name = match.group("name")
    if name not in TOOL_NAMES:
        raise RouteParseError(f"unknown tool {name!r}")
    args = _parse_escape_args(match.group("body"))
    return _decision_from_call(name, args, query, source, confidence)


def parse_json_tool_output(
    raw: str,
    *,
    query: str,
    source: RouteSource = "qwen",
    confidence: float = 0.68,
) -> RouteDecision:
    try:
        document = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise RouteParseError("router output is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != {"tool_calls"}:
        raise RouteParseError("router JSON must contain only tool_calls")
    calls = document["tool_calls"]
    if not isinstance(calls, list) or len(calls) != 1:
        raise RouteParseError("router must emit exactly one tool call")
    call = calls[0]
    if not isinstance(call, dict) or set(call) != {"name", "args"}:
        raise RouteParseError("tool call must contain exactly name and args")
    name = call["name"]
    args = call["args"]
    if name not in TOOL_NAMES:
        raise RouteParseError(f"unknown tool {name!r}")
    if not isinstance(args, dict):
        raise RouteParseError("tool args must be an object")
    return _decision_from_call(name, args, query, source, confidence)


def _parse_escape_args(body: str) -> dict[str, Any]:
    if not body.strip():
        return {}
    pattern = re.compile(r"(?P<key>[a-z_]+):<escape>(?P<value>.*?)<escape>", re.DOTALL)
    args: dict[str, Any] = {}
    cursor = 0
    for match in pattern.finditer(body):
        gap = body[cursor : match.start()]
        if gap.strip(" \t\r\n,"):
            raise RouteParseError("malformed FunctionGemma arguments")
        key = match.group("key")
        if key in args:
            raise RouteParseError(f"duplicate argument {key!r}")
        raw_value = match.group("value")
        if raw_value.startswith("[") or raw_value.startswith("{"):
            try:
                args[key] = json.loads(raw_value)
            except json.JSONDecodeError:
                args[key] = raw_value
        else:
            args[key] = raw_value
        cursor = match.end()
    if cursor == 0 or body[cursor:].strip(" \t\r\n,"):
        raise RouteParseError("malformed FunctionGemma arguments")
    return args


def _decision_from_call(
    name: str,
    args: dict[str, Any],
    query: str,
    source: RouteSource,
    confidence: float,
) -> RouteDecision:
    allowed: dict[str, set[str]] = {
        "store_memory": {"layer", "subject", "payload", "search_terms"},
        "retrieve_memory": {"layer", "subject", "search_terms"},
        "update_memory": {"layer", "subject", "payload", "search_terms"},
        "forget_memory": {"layer", "subject", "search_terms"},
        "search_memory": {"layer", "subject", "search_terms"},
        "query_code_graph": {"file_path", "symbol", "search_terms"},
        "answer_without_memory": set(),
        "escalate_external": set(),
    }
    required: dict[str, set[str]] = {
        "store_memory": {"layer", "subject", "payload"},
        "retrieve_memory": {"layer", "subject"},
        "update_memory": {"layer", "subject", "payload"},
        "forget_memory": {"layer", "subject"},
        "search_memory": {"layer", "search_terms"},
        "query_code_graph": set(),
        "answer_without_memory": set(),
        "escalate_external": set(),
    }
    extra = set(args) - allowed[name]
    if extra:
        raise RouteParseError(f"unexpected arguments for {name}: {sorted(extra)}")
    missing = required[name] - set(args)
    if missing:
        raise RouteParseError(f"missing arguments for {name}: {sorted(missing)}")

    operations: dict[str, RouteOperation] = {
        "store_memory": "store",
        "retrieve_memory": "retrieve",
        "update_memory": "update",
        "forget_memory": "forget",
        "search_memory": "search",
        "query_code_graph": "analyze_code",
        "answer_without_memory": "answer_without_memory",
        "escalate_external": "escalate_external",
    }
    operation = operations[name]
    layer_value: Any = "tree" if name == "query_code_graph" else args.get("layer", "none")
    if name in {"answer_without_memory", "escalate_external"}:
        layer_value = "none"
    if layer_value not in {"stm", "ltm", "tree", "none"}:
        raise RouteParseError(f"invalid memory layer: {layer_value!r}")
    layer = cast(RouteLayer, layer_value)
    search_terms = args.get("search_terms", [])
    if isinstance(search_terms, str):
        search_terms = [term for term in re.split(r"[,\s]+", search_terms) if term]
    if not isinstance(search_terms, list) or not all(
        isinstance(term, str) for term in search_terms
    ):
        raise RouteParseError("search_terms must be a list of strings")

    try:
        return RouteDecision(
            operation=operation,
            layer=layer,
            subject=args.get("subject"),
            payload=args.get("payload"),
            search_terms=search_terms,
            file_path=args.get("file_path"),
            symbol=args.get("symbol"),
            confidence=confidence,
            source=source,
        )
    except ValidationError as exc:
        raise RouteParseError(f"invalid {name} arguments: {exc}") from exc
