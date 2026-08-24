"""Public context result models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tst.scope.models import Scope


class ContextBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=2_000, ge=1, le=1_000_000)
    max_items: int = Field(default=24, ge=1, le=1_000)

    @classmethod
    def coerce(cls, value: ContextBudget | int | None) -> ContextBudget:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(max_tokens=int(value))


class ContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["memory", "tree"]
    scope: Scope
    key: str | None = None
    content: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=128)
    layer: Literal["stm", "ltm"] | None = None
    symbol: str | None = None
    file: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    project: str
    items: list[ContextItem] = Field(default_factory=list)
    estimated_tokens: int = Field(default=0, ge=0)

    def as_prompt(self) -> str:
        """Render a stable, provider-neutral prompt fragment."""

        if not self.items:
            return ""
        sections: list[str] = []
        current: Scope | None = None
        for item in self.items:
            if item.scope is not current:
                current = item.scope
                sections.append(f"{current.value.upper()} CONTEXT")
            location = item.file or item.symbol or item.key or item.source
            sections.append(f"- {location} ({item.reason}, {item.score:.2f})\n  {item.content}")
        return "\n".join(sections)
