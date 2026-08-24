"""Lightweight lexical retrieval and metadata reranking for v0.2."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from .planner import MemoryRecord

_GROUPS = (
    {"prefer", "preference", "preferred", "usually", "favorite", "favourite"},
    {"language", "programming", "coding"},
    {"editor", "ide"},
    {"service", "app", "application"},
    {"called", "name", "named"},
    {"recent", "just", "latest", "last"},
    {"use", "using", "used", "choose", "choice"},
    {"framework", "frontend"},
    {"test", "tests", "testing", "runner"},
    {"database", "db"},
    {"formatter", "formatting"},
    {"package", "packages", "manager", "install"},
    {"theme", "scheme", "terminal", "color", "colour"},
    {"deploy", "deployment", "region"},
    {"api", "style", "design"},
)
_SYNONYMS = {term: group for group in _GROUPS for term in group}
_CATEGORIES = (
    {"language", "programming", "coding"},
    {"editor", "ide"},
    {"framework", "frontend"},
    {"test", "tests", "testing", "runner"},
    {"database", "db"},
    {"formatter", "formatting"},
    {"package", "packages", "manager", "install"},
    {"theme", "scheme", "terminal", "color", "colour"},
    {"deploy", "deployment", "region"},
    {"api", "rest", "graphql"},
    {"service", "application", "app"},
)
_STOP = {
    "a",
    "an",
    "and",
    "do",
    "for",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "should",
    "the",
    "to",
    "what",
    "which",
    "you",
}


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    record: MemoryRecord
    score: float
    lexical_similarity: float
    key_prefix_match: float
    reinforcement: float
    recency: float


class LexicalMemoryRetriever:
    def __init__(self, *, min_score: float = 0.12) -> None:
        self.min_score = min_score

    def search(
        self,
        query: str,
        records: Iterable[MemoryRecord | dict],
        *,
        canonical_key: str | None = None,
        prefix: str | None = None,
        layer: Literal["stm", "ltm"] | None = None,
        top_k: int = 3,
        now_ms: int | None = None,
    ) -> list[ScoredMemory]:
        if top_k <= 0:
            return []
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        materialized = [
            record if isinstance(record, MemoryRecord) else MemoryRecord.model_validate(record)
            for record in records
        ]
        # Deduplicate by key and keep the newest non-deleted representation.
        latest: dict[str, MemoryRecord] = {}
        for record in materialized:
            if layer is not None and record.layer != layer:
                continue
            current = latest.get(record.key)
            if current is None or (record.updated_at, record.created_at) > (
                current.updated_at,
                current.created_at,
            ):
                latest[record.key] = record

        query_tokens = _expanded_tokens(query)
        results: list[ScoredMemory] = []
        for record in latest.values():
            if record.deleted:
                continue
            corpus_tokens = _expanded_tokens(_record_corpus(record))
            lexical_evidence = bool(query_tokens & corpus_tokens)
            prefix_only_query = not query_tokens and bool(prefix and record.key.startswith(prefix))
            if not lexical_evidence and record.key != canonical_key and not prefix_only_query:
                continue
            lexical = _lexical_score(query, query_tokens, record)
            key_match = _key_match(record.key, canonical_key, prefix)
            reinforcement = 1.0 - math.exp(-max(record.reinforcement_score, 0.0) / 10.0)
            recency = _recency(record.updated_at, now)
            score = (
                0.45 * lexical
                + 0.20 * key_match
                + 0.15 * reinforcement
                + 0.10 * recency
                + 0.10 * record.confidence
            )
            if score >= self.min_score:
                results.append(
                    ScoredMemory(
                        record=record,
                        score=min(score, 1.0),
                        lexical_similarity=lexical,
                        key_prefix_match=key_match,
                        reinforcement=reinforcement,
                        recency=recency,
                    )
                )
        results.sort(
            key=lambda item: (
                item.record.key == canonical_key,
                item.score,
                item.record.updated_at,
            ),
            reverse=True,
        )
        return results[:top_k]


def _tokens(text: str) -> set[str]:
    split = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    split = re.sub(r"[._:/-]+", " ", split.lower())
    return {
        token
        for token in re.findall(r"[a-z0-9+#.-]+", split)
        if token not in _STOP and len(token) > 1
    }


def _expanded_tokens(text: str) -> set[str]:
    result = _tokens(text)
    for token in tuple(result):
        result.update(_SYNONYMS.get(token, ()))
    return result


def _lexical_score(query: str, query_tokens: set[str], record: MemoryRecord) -> float:
    corpus = _record_corpus(record)
    record_tokens = _expanded_tokens(corpus)
    raw_query_tokens = _tokens(query)
    raw_record_tokens = _tokens(corpus)
    if not query_tokens or not record_tokens:
        overlap = 0.0
    else:
        overlap = len(query_tokens & record_tokens) / len(query_tokens | record_tokens)
        coverage = len(query_tokens & record_tokens) / len(query_tokens)
        overlap = 0.45 * overlap + 0.55 * coverage
    phrase = SequenceMatcher(None, query.lower(), corpus.lower()).ratio()
    category = max(
        (1.0 for terms in _CATEGORIES if raw_query_tokens & terms and raw_record_tokens & terms),
        default=0.0,
    )
    return min(1.0, 0.70 * overlap + 0.20 * category + 0.10 * phrase)


def _record_corpus(record: MemoryRecord) -> str:
    return " ".join([record.key.replace(":", " "), record.value, record.source_text, *record.tags])


def _key_match(key: str, canonical_key: str | None, prefix: str | None) -> float:
    if canonical_key and key == canonical_key:
        return 1.0
    if prefix and key.startswith(prefix):
        return 0.9
    if canonical_key:
        left = key.split(":")
        right = canonical_key.split(":")
        shared = 0
        for a, b in zip(left, right, strict=False):
            if a != b:
                break
            shared += 1
        return shared / max(len(left), len(right))
    return 0.0


def _recency(updated_at: int, now_ms: int) -> float:
    age_ms = max(0, now_ms - updated_at)
    # Seven-day half-life provides a weak tie-breaker, not dominant ranking.
    return math.exp(-math.log(2) * age_ms / (7 * 24 * 60 * 60 * 1000))


def memory_record_from_payload(
    payload: Any,
    *,
    key: str | None = None,
    layer: Literal["stm", "ltm"] | None = None,
) -> MemoryRecord | None:
    """Decode protocol and planner payload shapes into one strict record model."""

    if not isinstance(payload, dict):
        return None
    header: dict[str, Any] = payload["header"] if isinstance(payload.get("header"), dict) else {}
    data: Any = payload.get("data", payload)
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]
    if isinstance(data, dict):
        for variant in ("MemoryRecord", "memory_record"):
            if isinstance(data.get(variant), dict):
                data = data[variant]
                break
    if isinstance(data, dict) and "memory_type" in data:
        candidate = dict(data)
        # The Rust protocol annotates decoded records with its payload type;
        # that transport-only field is not part of the Python MemoryRecord.
        candidate.pop("payload_type", None)
        if key is not None:
            candidate.setdefault("key", key)
        if layer is not None:
            candidate.setdefault("layer", layer)
        try:
            return MemoryRecord.model_validate(candidate)
        except (TypeError, ValueError):
            return None

    preference: Any = None
    if isinstance(data, dict):
        preference = data.get("Preference") or data.get("preference")
        if preference is None and {"key", "value"} <= set(data):
            preference = data
    if not isinstance(preference, dict) or key is None or layer is None:
        return None
    created = _integer_timestamp(header.get("created_ts"))
    updated = max(created, _integer_timestamp(header.get("last_access_ts")))
    value = str(preference.get("value", ""))
    try:
        return MemoryRecord(
            key=key,
            value=value,
            memory_type="preference",
            source_text=value,
            created_at=created,
            updated_at=updated,
            confidence=1.0,
            tags=[str(preference.get("key", "preference"))],
            source="kernel",
            layer=layer,
        )
    except ValueError:
        return None


def _integer_timestamp(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], int):
        return max(0, value[0])
    return 0
