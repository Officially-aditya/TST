"""Tier orchestration: deterministic rules, FG, Qwen, then safe fallback."""

from __future__ import annotations

from collections.abc import Callable

from .decision import RouteDecision
from .deterministic import deterministic_route, mutation_is_authorized
from .parser import RouteParseError, parse_functiongemma_output, parse_json_tool_output

InferenceFunction = Callable[[str, str], str]


class ActionRouter:
    def __init__(
        self,
        functiongemma: InferenceFunction | None = None,
        qwen: InferenceFunction | None = None,
        *,
        external_provider_configured: bool = False,
    ) -> None:
        self.functiongemma = functiongemma
        self.qwen = qwen
        self.external_provider_configured = external_provider_configured
        self.parse_failures = 0
        self.inference_failures = 0
        self.tier_usage = {
            "deterministic": 0,
            "functiongemma": 0,
            "qwen": 0,
            "fallback": 0,
        }

    def route(self, query: str, payload: str = "") -> RouteDecision:
        direct = deterministic_route(query, payload or None)
        if direct is not None:
            self.tier_usage["deterministic"] += 1
            return direct

        if self.functiongemma is not None:
            try:
                raw = self.functiongemma(query, payload)
                decision = parse_functiongemma_output(raw, query=query)
                if self._is_allowed(query, decision):
                    self.tier_usage["functiongemma"] += 1
                    return decision
                self.parse_failures += 1
            except (RouteParseError, ValueError, TypeError):
                self.parse_failures += 1
            except Exception:
                self.inference_failures += 1

        if self.qwen is not None:
            try:
                raw = self.qwen(query, payload)
                decision = parse_json_tool_output(raw, query=query)
                if self._is_allowed(query, decision):
                    self.tier_usage["qwen"] += 1
                    return decision
                self.parse_failures += 1
            except (RouteParseError, ValueError, TypeError):
                self.parse_failures += 1
            except Exception:
                self.inference_failures += 1

        # A failed parse is never allowed to mutate memory.
        self.tier_usage["fallback"] += 1
        return self._fallback()

    def _is_allowed(self, query: str, decision: RouteDecision) -> bool:
        if decision.mutates_memory and not mutation_is_authorized(query, decision.operation):
            return False
        if decision.operation == "escalate_external" and not self.external_provider_configured:
            return False
        return True

    @staticmethod
    def _fallback() -> RouteDecision:
        return RouteDecision(
            operation="answer_without_memory",
            layer="none",
            subject=None,
            payload=None,
            search_terms=[],
            confidence=0.0,
            source="fallback",
        )
