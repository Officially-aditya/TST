"""The production memory retrieval and context-selection pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .context import ContextRanker, MemoryContext
from .planner import MemoryRecord
from .retrieval import LexicalMemoryRetriever, ScoredMemory


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    ranked: tuple[ScoredMemory, ...]
    context: MemoryContext


class RetrievalPipeline:
    def __init__(
        self,
        retriever: LexicalMemoryRetriever | None = None,
        ranker: ContextRanker | None = None,
    ) -> None:
        self.retriever = retriever or LexicalMemoryRetriever()
        self.ranker = ranker or ContextRanker()

    def retrieve(
        self,
        query: str,
        records: Iterable[MemoryRecord | dict],
        *,
        canonical_key: str | None = None,
        prefix: str | None = None,
        layer: Literal["stm", "ltm"] | None = None,
        top_k: int = 5,
        max_characters: int = 2_000,
        now_ms: int | None = None,
    ) -> RetrievalResult:
        ranked = self.retriever.search(
            query,
            records,
            canonical_key=canonical_key,
            prefix=prefix,
            layer=layer,
            top_k=top_k,
            now_ms=now_ms,
        )
        context = self.ranker.select(
            ranked,
            max_items=top_k,
            max_characters=max_characters,
        )
        return RetrievalResult(tuple(ranked), context)
