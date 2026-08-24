"""Serializable service response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tst.scope.models import Scope


class MemoryMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    scope: Scope
    layer: str
    value: str
    score: float = Field(ge=0.0, le=1.0)
    source_text: str = ""
    tags: list[str] = Field(default_factory=list)


class IntegrationStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: str
    location: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
