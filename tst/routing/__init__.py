"""Action-aware routing with deterministic intent handling and strict parsing."""

from .decision import RouteDecision
from .deterministic import deterministic_route
from .router import ActionRouter

__all__ = ["ActionRouter", "RouteDecision", "deterministic_route"]
