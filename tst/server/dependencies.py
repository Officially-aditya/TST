"""Dependency helpers kept separate so route tests can inject a service."""

from __future__ import annotations

from typing import Any


def service_from_request(request: Any) -> Any:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise RuntimeError("TST service is not initialized")
    return service
