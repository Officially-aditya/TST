"""Select ranked memory within an item and character budget."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .retrieval import ScoredMemory


@dataclass(frozen=True, slots=True)
class MemoryContext:
    text: str
    items: tuple[ScoredMemory, ...]
    characters: int


class ContextRanker:
    def select(
        self,
        ranked: Iterable[ScoredMemory],
        *,
        max_items: int = 5,
        max_characters: int = 2000,
    ) -> MemoryContext:
        if max_items <= 0 or max_characters <= 0:
            raise ValueError("context budgets must be positive")
        selected: list[ScoredMemory] = []
        lines: list[str] = []
        used = 0
        for item in ranked:
            if len(selected) >= max_items:
                break
            line = f"[{item.record.memory_type}] {item.record.value}"
            addition = len(line) + (1 if lines else 0)
            if used + addition > max_characters:
                continue
            selected.append(item)
            lines.append(line)
            used += addition
        return MemoryContext("\n".join(lines), tuple(selected), used)
