"""Deterministic context deduplication and budget selection."""

from __future__ import annotations

from collections.abc import Iterable

from .explain import estimate_tokens
from .models import ContextBudget, ContextItem


class ContextRanker:
    def select(
        self,
        items: Iterable[ContextItem],
        *,
        budget: ContextBudget | int | None = None,
    ) -> list[ContextItem]:
        limits = ContextBudget.coerce(budget)
        unique: dict[str, ContextItem] = {}
        for item in items:
            identity = item.key or f"{item.source}:{item.file}:{item.symbol}:{item.content}"
            existing = unique.get(identity)
            if existing is None or item.score > existing.score:
                unique[identity] = item
        ordered = sorted(
            unique.values(),
            key=lambda item: (-item.score, item.scope.value, item.key or item.content),
        )
        selected: list[ContextItem] = []
        used = 0
        for item in ordered:
            if len(selected) >= limits.max_items:
                break
            cost = estimate_tokens(item.content)
            if selected and used + cost > limits.max_tokens:
                continue
            selected.append(item)
            used += cost
        return selected
