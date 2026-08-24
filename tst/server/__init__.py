"""Optional local HTTP server for the TST control plane."""

from __future__ import annotations

from typing import Any

__all__ = ["create_app", "run"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .app import create_app, run

        return {"create_app": create_app, "run": run}[name]
    raise AttributeError(name)
