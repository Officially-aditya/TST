"""Turn route decisions into protocol operations and metadata-rich payloads."""

from __future__ import annotations

import re
import time
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tst.protocol.operations import ProtocolOperation
from tst.routing.decision import RouteDecision

from .keys import CanonicalKeyBuilder


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: str
    memory_type: str = Field(min_length=1, max_length=128)
    source_text: str
    created_at: int = Field(ge=0)
    updated_at: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    source: str = Field(default="user", min_length=1, max_length=128)
    layer: Literal["stm", "ltm"] = "ltm"
    reinforcement_score: float = Field(default=0.0, ge=0.0)
    deleted: bool = False

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return CanonicalKeyBuilder.validate(value)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for tag in value:
            cleaned = tag.strip().lower()
            if not cleaned or len(cleaned) > 128:
                raise ValueError("tags must be non-empty and at most 128 characters")
            if cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    @model_validator(mode="after")
    def validate_timestamps(self) -> MemoryRecord:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        return self


class MemoryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_operation: ProtocolOperation | None
    params: dict[str, Any] = Field(default_factory=dict)
    canonical_key: str | None = None
    retrieval_query: str | None = None


class MemoryPlanner:
    def __init__(self, keys: CanonicalKeyBuilder | None = None) -> None:
        self.keys = keys or CanonicalKeyBuilder()

    def plan(self, decision: RouteDecision, user_query: str) -> MemoryPlan:
        if decision.operation in {"answer_without_memory", "escalate_external"}:
            return MemoryPlan(protocol_operation=None)

        if decision.operation == "analyze_code":
            name = decision.symbol or decision.file_path or decision.subject or user_query
            return MemoryPlan(
                protocol_operation=ProtocolOperation.TREE_FIND,
                params={"name": name, "limit": 20},
                retrieval_query=name,
            )

        if decision.layer not in {"stm", "ltm"}:
            raise ValueError("memory plans require the stm or ltm layer")
        memory_layer = cast(Literal["stm", "ltm"], decision.layer)

        memory_type = _memory_type(decision.subject, decision.payload or user_query)
        subject = decision.subject or _subject_from_text(decision.payload or user_query)
        key = self.keys.for_subject(memory_layer, subject, memory_type=memory_type)

        if decision.operation in {"store", "update"}:
            assert decision.payload is not None
            now = int(time.time() * 1000)
            record = MemoryRecord(
                key=key,
                value=_extract_value(decision.payload),
                memory_type=memory_type,
                source_text=decision.payload,
                created_at=now,
                updated_at=now,
                confidence=decision.confidence,
                tags=_tags(subject, decision.payload, decision.search_terms),
                source="user",
                layer=memory_layer,
                reinforcement_score=0.0,
            )
            payload_type = "preference" if memory_type == "preference" else "token_stats"
            operation = (
                ProtocolOperation.MEMORY_STORE
                if decision.operation == "store"
                else ProtocolOperation.MEMORY_UPDATE
            )
            return MemoryPlan(
                protocol_operation=operation,
                params={
                    "layer": memory_layer,
                    "key": key,
                    "payload": {"type": payload_type, "data": record.model_dump()},
                },
                canonical_key=key,
            )

        if decision.operation == "forget":
            return MemoryPlan(
                protocol_operation=ProtocolOperation.MEMORY_DELETE,
                params={"layer": memory_layer, "key": key},
                canonical_key=key,
            )

        search_query = " ".join(
            part for part in [decision.subject or "", *decision.search_terms, user_query] if part
        )
        if decision.operation == "retrieve" and decision.subject:
            return MemoryPlan(
                protocol_operation=ProtocolOperation.MEMORY_GET,
                params={"layer": memory_layer, "key": key},
                canonical_key=key,
                retrieval_query=search_query,
            )
        return MemoryPlan(
            protocol_operation=ProtocolOperation.MEMORY_SEARCH,
            params={"layer": memory_layer, "query": search_query, "limit": 10},
            canonical_key=key,
            retrieval_query=search_query,
        )


def _memory_type(subject: str | None, text: str) -> str:
    value = f"{subject or ''} {text}".lower()
    if subject in {"programming_language", "editor", "theme", "framework"}:
        return "preference"
    if re.search(r"\b(?:prefer|preference|always use|favorite|favourite)\b", value):
        return "preference"
    return "context" if subject in {"recent_context", "service_name"} else "fact"


def _subject_from_text(text: str) -> str:
    lowered = text.lower()
    categories = (
        (r"\b(?:language|typescript|javascript|python|rust|java)\b", "programming_language"),
        (r"\b(?:editor|ide|vim|neovim|emacs|vscode)\b", "editor"),
        (r"\b(?:theme|dark mode|light mode)\b", "theme"),
        (r"\bservice\b", "service_name"),
        (r"\bframework\b", "framework"),
    )
    for pattern, subject in categories:
        if re.search(pattern, lowered):
            return subject
    terms = re.findall(r"[a-z0-9_+#.-]+", lowered)
    return "_".join(terms[:6]) or "general"


def _extract_value(text: str) -> str:
    patterns = (
        r"\bprefer\s+(.+?)(?:\s+over\s+.+)?$",
        r"\b(?:always\s+)?use\s+(.+?)(?:\s+over\s+.+)?$",
        r"\bcall\s+the\s+service\s+(.+)$",
        r"\bservice\s+(?:is|called)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text.strip().rstrip("."), re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()


def _tags(subject: str, payload: str, supplied: list[str]) -> list[str]:
    raw = [subject, *supplied, *re.findall(r"[A-Za-z0-9_+#.-]+", payload)]
    normalized = [term.lower().replace("_", "-") for term in raw if len(term) > 1]
    return list(dict.fromkeys(normalized))[:24]
