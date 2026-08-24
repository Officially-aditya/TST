"""Command registry value types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    name: str
    args: tuple[str, ...] = ()
    raw: str = ""


@dataclass(frozen=True, slots=True)
class CommandResult:
    output: str = ""
    data: Any = None


CommandHandler = Callable[[ParsedCommand], CommandResult]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = field(default_factory=tuple)
