"""TST local-memory framework.

The public package is intentionally import-light. Model runtimes and FastAPI are
loaded only by their respective entry points; protocol, routing, and retrieval
utilities can be used without Torch or Transformers installed.
"""

from typing import Any

__all__ = ["RouteDecision", "StdioKernelClient"]
__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    """Load optional Pydantic-backed public types only when requested."""

    if name == "StdioKernelClient":
        from .kernel.client import StdioKernelClient

        return StdioKernelClient
    if name == "RouteDecision":
        from .routing.decision import RouteDecision

        return RouteDecision
    raise AttributeError(name)
