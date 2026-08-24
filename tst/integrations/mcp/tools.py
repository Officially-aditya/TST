"""MCP tool dispatch with no direct kernel access."""

from __future__ import annotations

from typing import Any

from tst.scope.models import Scope
from tst.service.service import TSTService

from .schemas import TOOL_DEFINITIONS


def tool_definitions() -> list[dict[str, object]]:
    return [dict(item) for item in TOOL_DEFINITIONS]


def call_tool(service: TSTService, name: str, arguments: dict[str, Any] | None = None) -> Any:
    args = arguments or {}
    if name == "tst_status":
        return service.status()
    if name == "tst_context":
        return service.retrieve_context(args["query"], budget=args.get("budget", 2_000)).model_dump(mode="json")
    if name == "tst_memory_search":
        scope = Scope.coerce(args["scope"]) if args.get("scope") else None
        return [
            match.model_dump(mode="json")
            for match in service.search_memories(args["query"], scope, limit=args.get("limit", 20))
        ]
    if name == "tst_memory_store":
        payload = dict(args)
        if "scope" in payload:
            payload["scope"] = Scope.coerce(payload["scope"])
        return service.store_memory(**payload)
    if name == "tst_memory_update":
        payload = dict(args)
        key = str(payload.pop("key"))
        if "scope" in payload and payload["scope"] is not None:
            payload["scope"] = Scope.coerce(payload["scope"])
        return service.update_memory(key, **payload)
    if name == "tst_memory_forget":
        scope = Scope.coerce(args["scope"]) if args.get("scope") else None
        return service.forget_memory(str(args["key"]), scope=scope)
    if name == "tst_tree_find":
        return service.find_symbol(str(args["name"]), limit=args.get("limit", 20))
    if name == "tst_tree_query":
        payload = dict(args)
        symbol = payload.pop("symbol")
        return service.query_symbol(symbol, **payload)
    raise ValueError(f"unknown TST MCP tool: {name}")
