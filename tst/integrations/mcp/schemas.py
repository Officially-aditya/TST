"""MCP tool schemas kept independent from any MCP SDK version."""

from __future__ import annotations

TOOL_DEFINITIONS: list[dict[str, object]] = [
    {
        "name": "tst_status",
        "description": "Return TST project and local kernel health.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "tst_context",
        "description": "Retrieve bounded, explainable context across TST scopes and the project tree.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "budget": {"type": "integer", "minimum": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tst_memory_search",
        "description": "Search TST memories by query and optional scope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "scope": {"type": "string", "enum": ["global", "project", "session"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tst_memory_store",
        "description": "Store an explicitly requested memory in a chosen scope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "scope": {"type": "string", "enum": ["global", "project", "session"]},
                "key": {"type": "string"},
                "subject": {"type": "string"},
                "memory_type": {"type": "string"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tst_memory_update",
        "description": "Update an existing TST memory.",
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}, "scope": {"type": "string"}},
            "required": ["key", "value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tst_memory_forget",
        "description": "Forget one TST memory by canonical key.",
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "scope": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tst_tree_find",
        "description": "Find project-local symbols or files.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tst_tree_query",
        "description": "Query a bounded project-local symbol neighborhood.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": ["string", "integer"]},
                "depth": {"type": "integer", "minimum": 0, "maximum": 32},
                "max_nodes": {"type": "integer", "minimum": 1, "maximum": 10000},
                "token_budget": {"type": "integer", "minimum": 1, "maximum": 1000000},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
]
