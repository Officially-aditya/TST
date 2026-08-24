"""Shared policy and rendering for automatic agent context retrieval."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tst.context.formatting import format_context_document

DEFAULT_BUDGET = 2_000
DEFAULT_TIMEOUT = 20
DEFAULT_MAX_QUERY_CHARS = 16_000


def automatic_context_enabled(environment: Mapping[str, str] | None = None) -> bool:
    values = environment if environment is not None else os.environ
    explicit_override = values.get("TST_CONTEXT_AUTO")
    if explicit_override is not None and explicit_override.strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    mode = values.get("TST_CONTEXT_MODE", "auto").strip().lower()
    return mode in {"auto", "on", "true", "1"}


def context_budget(environment: Mapping[str, str] | None = None) -> int:
    values = environment if environment is not None else os.environ
    return _bounded_integer(values.get("TST_CONTEXT_BUDGET"), DEFAULT_BUDGET, minimum=1, maximum=1_000_000)


def context_timeout(environment: Mapping[str, str] | None = None) -> int:
    values = environment if environment is not None else os.environ
    return _bounded_integer(values.get("TST_CONTEXT_TIMEOUT"), DEFAULT_TIMEOUT, minimum=1, maximum=120)


def context_query(prompt: str, environment: Mapping[str, str] | None = None) -> str:
    values = environment if environment is not None else os.environ
    maximum = _bounded_integer(
        values.get("TST_CONTEXT_MAX_QUERY_CHARS"),
        DEFAULT_MAX_QUERY_CHARS,
        minimum=256,
        maximum=100_000,
    )
    return prompt.strip()[:maximum]


def retrieve_context(
    prompt: str,
    project: str | Path,
    *,
    actor: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Retrieve and render context without logging or persisting the prompt."""

    values = environment if environment is not None else os.environ
    if not automatic_context_enabled(values):
        return ""
    query = context_query(prompt, values)
    if not query:
        return ""
    command = values.get("TST_BIN", "tst")
    arguments = [
        command,
        "context",
        "--project",
        str(Path(project).expanduser().resolve()),
        "--query",
        query,
        "--budget",
        str(context_budget(values)),
        "--actor",
        actor,
        "--json",
    ]
    try:
        result = subprocess.run(
            arguments,
            cwd=Path(project).expanduser().resolve(),
            capture_output=True,
            check=False,
            text=True,
            timeout=context_timeout(values),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    try:
        document = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return ""
    return render_context_document(document)


def render_context_document(document: Mapping[str, Any]) -> str:
    """Render context as readable Markdown without exposing retrieval internals."""

    return format_context_document(document)


def _bounded_integer(value: str | None, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))
