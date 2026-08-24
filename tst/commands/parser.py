"""Shell-like parser for TST commands and autocomplete suggestions."""

from __future__ import annotations

import shlex
from collections.abc import Iterable

from .models import CommandSpec, ParsedCommand


def parse_command(text: str) -> ParsedCommand:
    if not text.lstrip().startswith("/"):
        raise ValueError("TST commands must start with '/'")
    try:
        tokens = shlex.split(text.strip()[1:])
    except ValueError as exc:
        raise ValueError(f"invalid command quoting: {exc}") from exc
    if not tokens or not tokens[0]:
        raise ValueError("command name cannot be empty")
    return ParsedCommand(tokens[0].casefold(), tuple(tokens[1:]), text)


def suggest_commands(prefix: str, specs: Iterable[CommandSpec]) -> list[CommandSpec]:
    query = prefix.removeprefix("/").casefold()
    return sorted(
        [
            spec
            for spec in specs
            if spec.name.casefold().startswith(query)
            or any(alias.casefold().startswith(query) for alias in spec.aliases)
        ],
        key=lambda spec: spec.name,
    )
