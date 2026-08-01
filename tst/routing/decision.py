"""The operation-and-layer decision shared by every router tier."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RouteOperation = Literal[
    "store",
    "retrieve",
    "update",
    "forget",
    "search",
    "analyze_code",
    "answer_without_memory",
    "escalate_external",
]
RouteLayer = Literal["stm", "ltm", "tree", "none"]
RouteSource = Literal["deterministic", "functiongemma", "qwen", "fallback"]


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: RouteOperation
    layer: RouteLayer
    subject: str | None = None
    payload: str | None = None
    search_terms: list[str] = Field(default_factory=list)
    file_path: str | None = None
    symbol: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: RouteSource

    @model_validator(mode="after")
    def _operation_matches_layer(self) -> RouteDecision:
        if self.operation == "analyze_code" and self.layer != "tree":
            raise ValueError("analyze_code must target the tree layer")
        if self.operation in {"answer_without_memory", "escalate_external"}:
            if self.layer != "none":
                raise ValueError(f"{self.operation} must use the none layer")
        elif self.operation != "analyze_code" and self.layer not in {"stm", "ltm"}:
            raise ValueError(f"{self.operation} must target stm or ltm")
        elif self.layer == "none":
            raise ValueError(f"{self.operation} requires a memory layer")
        if self.operation in {"store", "update"} and not self.payload:
            raise ValueError(f"{self.operation} requires a non-empty payload")
        if self.operation == "forget" and not (self.subject or self.search_terms):
            raise ValueError("forget requires a subject or search terms")
        return self

    @property
    def mutates_memory(self) -> bool:
        return self.operation in {"store", "update", "forget"}
