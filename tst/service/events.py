"""Bounded in-memory activity stream with conservative metadata redaction."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from tst.scope.models import Scope


class ActivityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: float = Field(default_factory=time.time, ge=0)
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    actor: str = "TST"
    operation: str
    scope: Scope | None = None
    duration_ms: float = Field(default=0.0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Keep recent events and notify local subscribers without persistence."""

    def __init__(self, *, max_events: int = 500) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self._events: deque[ActivityEvent] = deque(maxlen=max_events)
        self._subscribers: set[queue.Queue[ActivityEvent]] = set()
        self._lock = threading.RLock()

    def publish(
        self,
        operation: str,
        *,
        actor: str = "TST",
        scope: Scope | str | None = None,
        request_id: str | None = None,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ActivityEvent:
        event = ActivityEvent(
            operation=operation,
            actor=actor,
            scope=Scope.coerce(scope) if scope is not None else None,
            request_id=request_id or uuid4().hex,
            duration_ms=max(0.0, duration_ms),
            metadata=redact_metadata(metadata or {}),
        )
        with self._lock:
            self._events.append(event)
            for subscriber in tuple(self._subscribers):
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    # A slow UI must not block a memory operation.
                    pass
        return event

    def snapshot(self) -> list[ActivityEvent]:
        with self._lock:
            return list(self._events)

    def subscribe(self, *, maxsize: int = 100) -> queue.Queue[ActivityEvent]:
        subscriber: queue.Queue[ActivityEvent] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[ActivityEvent]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def iter_subscriber(self, subscriber: queue.Queue[ActivityEvent], timeout: float = 30.0) -> Iterator[ActivityEvent]:
        try:
            while True:
                try:
                    yield subscriber.get(timeout=timeout)
                except queue.Empty:
                    return
        finally:
            self.unsubscribe(subscriber)


_SENSITIVE_KEYS = {
    "content",
    "memory",
    "payload",
    "prompt",
    "query",
    "source_text",
    "value",
}


def redact_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Remove prompt and memory contents while preserving useful counters."""

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key.casefold() in _SENSITIVE_KEYS:
            if isinstance(item, str):
                result[f"{key}_length"] = len(item)
            continue
        if isinstance(item, dict):
            result[key] = redact_metadata(item)
        elif isinstance(item, list):
            if all(isinstance(entry, dict) for entry in item):
                result[key] = [redact_metadata(entry) for entry in item]
            else:
                result[key] = {"count": len(item)}
        else:
            result[key] = item
    return result
