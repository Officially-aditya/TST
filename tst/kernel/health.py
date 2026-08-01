"""Small typed health view over ``kernel.status``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .client import StdioKernelClient


@dataclass(frozen=True, slots=True)
class KernelHealth:
    running: bool
    status: dict[str, Any] | None
    error: str | None = None


def check_kernel(client: StdioKernelClient) -> KernelHealth:
    if not client.is_running:
        return KernelHealth(running=False, status=None, error="process is not running")
    try:
        return KernelHealth(running=True, status=client.status())
    except Exception as exc:
        return KernelHealth(running=False, status=None, error=str(exc))
