"""Reusable TST slash-command parsing and dispatch."""

from .models import CommandResult, CommandSpec
from .parser import ParsedCommand, parse_command, suggest_commands
from .registry import CommandRegistry, default_registry

__all__ = [
    "CommandRegistry",
    "CommandResult",
    "CommandSpec",
    "ParsedCommand",
    "default_registry",
    "parse_command",
    "suggest_commands",
]
