"""Compact action-aware tool schemas used by both model router tiers."""

from __future__ import annotations

_LAYER = {"type": "string", "enum": ["stm", "ltm"]}
_TERMS = {"type": "array", "items": {"type": "string"}}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOL_SCHEMAS = [
    _tool(
        "store_memory",
        "Store new information only when the user explicitly asks to remember it.",
        {
            "layer": _LAYER,
            "subject": {"type": "string"},
            "payload": {"type": "string"},
            "search_terms": _TERMS,
        },
        ["layer", "subject", "payload"],
    ),
    _tool(
        "retrieve_memory",
        "Retrieve previously stored information without creating or changing memory.",
        {"layer": _LAYER, "subject": {"type": "string"}, "search_terms": _TERMS},
        ["layer", "subject"],
    ),
    _tool(
        "update_memory",
        "Replace a specific previously stored memory.",
        {
            "layer": _LAYER,
            "subject": {"type": "string"},
            "payload": {"type": "string"},
            "search_terms": _TERMS,
        },
        ["layer", "subject", "payload"],
    ),
    _tool(
        "forget_memory",
        "Delete a specific memory only when the user explicitly asks to forget it.",
        {"layer": _LAYER, "subject": {"type": "string"}, "search_terms": _TERMS},
        ["layer", "subject"],
    ),
    _tool(
        "search_memory",
        "Search a memory category when an exact subject is not known.",
        {"layer": _LAYER, "subject": {"type": "string"}, "search_terms": _TERMS},
        ["layer", "search_terms"],
    ),
    _tool(
        "query_code_graph",
        "Analyze a concrete source path or query a repository symbol graph.",
        {"file_path": {"type": "string"}, "symbol": {"type": "string"}, "search_terms": _TERMS},
        [],
    ),
    _tool("answer_without_memory", "Answer locally without reading or mutating memory.", {}, []),
    _tool(
        "escalate_external",
        "Use only when an explicitly configured external provider is required.",
        {},
        [],
    ),
]

FG_TOOL_SCHEMAS = [{"type": "function", "function": schema} for schema in TOOL_SCHEMAS]
VALID_TOOLS = {schema["name"] for schema in TOOL_SCHEMAS}
