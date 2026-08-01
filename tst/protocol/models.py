"""Pydantic request and response envelopes for the NDJSON protocol."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .operations import ProtocolOperation

PROTOCOL_VERSION = 1


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KernelRequest(_StrictModel):
    protocol_version: int = Field(default=PROTOCOL_VERSION, ge=1)
    request_id: str = Field(min_length=1, max_length=128)
    operation: ProtocolOperation
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _supported_version(self) -> KernelRequest:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol version {self.protocol_version}; expected {PROTOCOL_VERSION}"
            )
        return self

    @classmethod
    def create(
        cls,
        operation: ProtocolOperation | str,
        params: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> KernelRequest:
        return cls(
            request_id=request_id or uuid4().hex,
            operation=ProtocolOperation(operation),
            params=params or {},
        )


class KernelErrorData(_StrictModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
    retryable: bool = False


class KernelMetrics(_StrictModel):
    kernel_ms: float | None = Field(default=None, ge=0)


class KernelResponse(_StrictModel):
    protocol_version: int
    request_id: str
    ok: bool
    result: dict[str, Any] | None = None
    error: KernelErrorData | None = None
    metrics: KernelMetrics | None = None

    @model_validator(mode="after")
    def _consistent_outcome(self) -> KernelResponse:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol version {self.protocol_version}; expected {PROTOCOL_VERSION}"
            )
        if self.ok and self.error is not None:
            raise ValueError("successful response cannot contain an error")
        if not self.ok and self.error is None:
            raise ValueError("failed response must contain an error")
        return self
