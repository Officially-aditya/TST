"""Small helpers for human-readable retrieval provenance."""

from __future__ import annotations

import math
import re


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def memory_reason(query: str, key: str, content: str, *, prefix_match: bool = False) -> str:
    if prefix_match and not query.strip():
        return "scope_match"
    query_terms = _terms(query)
    corpus = _terms(f"{key} {content}")
    if query_terms and query_terms <= corpus:
        return "lexical_match"
    return "related_memory"


def tree_reason(query: str, qualified_name: str, *, related: bool = False) -> str:
    if related:
        return "relationship_match"
    if qualified_name.casefold() == query.casefold().strip():
        return "symbol_match"
    return "name_match"


def _terms(value: str) -> set[str]:
    split = re.sub(r"([a-z])([A-Z])", r"\1 \2", value.casefold())
    return {term for term in re.findall(r"[a-z0-9_+#.-]+", split) if len(term) > 1}
