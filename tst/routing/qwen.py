"""Qwen routing prompt and strict parser re-export."""

from .parser import parse_json_tool_output

SYSTEM_PROMPT = (
    "You are a memory action router. Call exactly one available tool. "
    "Return only a JSON object with one tool_calls entry and no prose."
)

__all__ = ["SYSTEM_PROMPT", "parse_json_tool_output"]
