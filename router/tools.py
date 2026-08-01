"""Protocol-neutral action handlers for router decisions.

This module contains no HTTP kernel assumptions. CLI, FastAPI, and evaluations
all pass the same :class:`~tst.kernel.client.StdioKernelClient` instance here.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from tst.kernel.client import StdioKernelClient
from tst.memory.pipeline import RetrievalPipeline
from tst.memory.planner import MemoryPlan, MemoryPlanner
from tst.memory.retrieval import memory_record_from_payload
from tst.protocol.models import KernelResponse
from tst.protocol.operations import ProtocolOperation
from tst.routing.decision import RouteDecision
from tst.routing.schemas import FG_TOOL_SCHEMAS, TOOL_SCHEMAS, VALID_TOOLS

# Compatibility export name used by evaluation scripts. Values are action-tool
# names now; legacy route_to_* names are intentionally not accepted.
VALID_ROUTES = VALID_TOOLS


class ActionHandlers:
    def __init__(
        self,
        client: StdioKernelClient | None,
        *,
        planner: MemoryPlanner | None = None,
        retrieval: RetrievalPipeline | None = None,
        external_provider_configured: bool = False,
        context_max_items: int = 5,
        context_max_characters: int = 2_000,
    ) -> None:
        if context_max_items <= 0 or context_max_characters <= 0:
            raise ValueError("context budgets must be positive")
        self.client = client
        self.planner = planner or MemoryPlanner()
        self.retrieval = retrieval or RetrievalPipeline()
        self.external_provider_configured = external_provider_configured
        self.context_max_items = context_max_items
        self.context_max_characters = context_max_characters

    def dispatch(self, decision: RouteDecision, query: str) -> dict[str, Any]:
        if decision.operation == "answer_without_memory":
            return {
                "operation": decision.operation,
                "layer": "none",
                "kernel": None,
                "context": [],
                "escalate": False,
            }
        if decision.operation == "escalate_external":
            if not self.external_provider_configured:
                return {
                    "operation": "answer_without_memory",
                    "requested_operation": decision.operation,
                    "layer": "none",
                    "kernel": None,
                    "context": [],
                    "escalate": False,
                    "external_configured": False,
                }
            return {
                "operation": decision.operation,
                "layer": "none",
                "kernel": None,
                "context": [],
                "escalate": self.external_provider_configured,
                "external_configured": self.external_provider_configured,
            }
        if self.client is None:
            return {
                "operation": decision.operation,
                "layer": decision.layer,
                "kernel": None,
                "context": [],
                "error": "kernel unavailable",
                "escalate": False,
            }

        plan = self.planner.plan(decision, query)
        if plan.protocol_operation is None:
            return {
                "operation": decision.operation,
                "layer": decision.layer,
                "kernel": None,
                "context": [],
                "escalate": False,
            }

        response = self.client.request(plan.protocol_operation, plan.params)
        result = response.result or {}

        if decision.operation in {"retrieve", "search"}:
            return self._retrieval_result(decision, plan, response, result)

        return {
            "operation": decision.operation,
            "layer": decision.layer,
            "key": plan.canonical_key,
            "kernel": response.model_dump(mode="json"),
            "context": _context_items(result),
            "escalate": False,
        }

    def _retrieval_result(
        self,
        decision: RouteDecision,
        plan: MemoryPlan,
        response: KernelResponse,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        assert self.client is not None
        layer = decision.layer
        if layer not in {"stm", "ltm"}:
            raise ValueError("memory retrieval requires stm or ltm")
        memory_layer = cast(Literal["stm", "ltm"], layer)

        records = _records_from_result(result, memory_layer)
        exact_decoded = bool(records)
        exact_found = bool(result.get("found"))
        prefix = _category_prefix(plan.canonical_key)

        # Exact lookup is authoritative when it yields a valid live record.
        # Otherwise gather bounded category and lexical candidate sets before
        # applying the same weighted pipeline used by evaluation.
        if not records or not exact_found:
            if prefix:
                prefix_response = self.client.request(
                    ProtocolOperation.MEMORY_SEARCH,
                    {"layer": memory_layer, "query": "", "prefix": prefix, "limit": 100},
                )
                records.extend(_records_from_result(prefix_response.result or {}, memory_layer))
                response = prefix_response
            if plan.retrieval_query:
                lexical_response = self.client.request(
                    ProtocolOperation.MEMORY_SEARCH,
                    {"layer": memory_layer, "query": plan.retrieval_query, "limit": 100},
                )
                records.extend(_records_from_result(lexical_response.result or {}, memory_layer))
                response = lexical_response

        retrieval = self.retrieval.retrieve(
            plan.retrieval_query or "",
            records,
            canonical_key=plan.canonical_key,
            prefix=prefix,
            layer=memory_layer,
            top_k=self.context_max_items,
            max_characters=self.context_max_characters,
        )
        context = [item.record.value for item in retrieval.context.items]
        if not context and exact_found and not exact_decoded:
            context = _context_items(result)
        return {
            "operation": decision.operation,
            "layer": memory_layer,
            "key": plan.canonical_key,
            "kernel": response.model_dump(mode="json"),
            "context": context,
            "context_text": retrieval.context.text,
            "matches": [
                {
                    "key": item.record.key,
                    "layer": item.record.layer,
                    "score": item.score,
                    "value": item.record.value,
                }
                for item in retrieval.ranked
            ],
            "escalate": False,
        }


def dispatch_decision(
    client: StdioKernelClient | None,
    decision: RouteDecision,
    query: str,
    *,
    external_provider_configured: bool = False,
) -> dict[str, Any]:
    return ActionHandlers(
        client,
        external_provider_configured=external_provider_configured,
    ).dispatch(decision, query)


def _context_items(result: dict[str, Any]) -> list[str]:
    candidates: list[Any]
    if result.get("found") and result.get("payload") is not None:
        candidates = [result["payload"]]
    elif isinstance(result.get("matches"), list):
        candidates = [
            match.get("payload") for match in result["matches"] if isinstance(match, dict)
        ]
    elif isinstance(result.get("nodes"), list):
        return [
            f"{node.get('node_type', 'node')}:{node.get('name', '')}"
            for node in result["nodes"]
            if isinstance(node, dict)
        ]
    else:
        return []

    values: list[str] = []
    for payload in candidates:
        value = _payload_value(payload)
        if value and value not in values:
            values.append(value)
    return values


def _records_from_result(result: dict[str, Any], layer: Literal["stm", "ltm"]) -> list[Any]:
    candidates: list[tuple[str | None, Any]] = []
    if result.get("found") and result.get("payload") is not None:
        candidates.append((str(result.get("key", "")) or None, result["payload"]))
    matches = result.get("matches")
    if isinstance(matches, list):
        for match in matches:
            if not isinstance(match, dict) or match.get("layer", layer) != layer:
                continue
            candidates.append((str(match.get("key", "")) or None, match.get("payload")))
    records: list[Any] = []
    for key, payload in candidates:
        if key is None:
            continue
        record = memory_record_from_payload(payload, key=key, layer=layer)
        if record is not None:
            records.append(record)
    return records


def _category_prefix(canonical_key: str | None) -> str | None:
    if not canonical_key or ":" not in canonical_key:
        return None
    return canonical_key.rsplit(":", 1)[0] + ":"


def _payload_value(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return str(payload) if payload is not None else None
    data = payload.get("data", payload)
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        # Plan-style payload: {type, data: MemoryRecord}.
        data = data["data"]
    if isinstance(data, dict) and isinstance(data.get("value"), str):
        return data["value"]
    if isinstance(data, dict) and isinstance(data.get("source_text"), str):
        return data["source_text"]
    if isinstance(data, dict) and isinstance(data.get("MemoryRecord"), dict):
        record = data["MemoryRecord"]
        value = record.get("value") or record.get("source_text")
        return str(value) if value is not None else None
    if isinstance(data, dict) and isinstance(data.get("Preference"), dict):
        value = data["Preference"].get("value")
        return str(value) if value is not None else None
    if isinstance(data, dict) and isinstance(data.get("TokenStats"), dict):
        value = data["TokenStats"].get("canonical_form")
        return str(value) if value is not None else None
    return None


__all__ = [
    "ActionHandlers",
    "FG_TOOL_SCHEMAS",
    "TOOL_SCHEMAS",
    "VALID_ROUTES",
    "VALID_TOOLS",
    "dispatch_decision",
]
