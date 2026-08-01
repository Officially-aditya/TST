"""Typed failures raised by the shared kernel client."""

from __future__ import annotations

from typing import Any


class KernelClientError(RuntimeError):
    """Base class for Python-to-kernel failures."""


class KernelNotStartedError(KernelClientError):
    """The client was used before a live process was available."""


class KernelStartError(KernelClientError):
    """The kernel binary could not be started or did not become ready."""


class KernelExitedError(KernelClientError):
    """The kernel process exited while a request was in flight."""


class KernelTimeoutError(KernelClientError):
    """A startup or request deadline elapsed."""


class KernelProtocolError(KernelClientError):
    """The kernel emitted invalid or mismatched protocol data."""


class KernelOperationError(KernelClientError):
    """The kernel returned a structured operation error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        request_id: str,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.request_id = request_id
        self.details = details
        self.retryable = retryable
