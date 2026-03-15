"""
router/tools.py

Tool schema definitions and dispatch functions for the TST Memory Router.

Two layers:
  1. TOOL_SCHEMAS  — plain JSON dicts embedded into the inference prompt so
                     FunctionGemma / Qwen3 can decide which tool to call.
  2. dispatch_*()  — Python functions executed after the routing decision,
                     sending the correct JSON payload to the Rust Kernel.

Kernel interface
----------------
The Rust Kernel (tst_memory) listens on port 3000 (HTTP) and accepts JSON
bodies shaped as:

  WRITE:
    POST /write
    {
      "op": "insert",
      "key": "<string>",
      "payload": {
        "header": {
          "payload_type": <int>,   // 1=TokenStats, 2=CodeNode, 3=Relation
          "version": 1,
          "created_ts":     {"Timestamp": <unix_ms>},
          "last_access_ts": {"Timestamp": <unix_ms>},
          "access_count": 1
        },
        "data": { ... }
      }
    }

  READ:
    POST /read
    {"keys": ["<key>"], "max_results": <int>}

Cloud escalation does NOT write to the kernel — it returns a sentinel so
the caller knows to forward the query externally.
"""

import json
import time
import hashlib
import httpx
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KERNEL_URL = "http://127.0.0.1:3000"
HTTP_TIMEOUT = 5.0  # seconds

# payload_type codes (must match Rust PayloadType enum)
PAYLOAD_TYPE_TOKEN_STATS = 1
PAYLOAD_TYPE_CODE_NODE   = 2
PAYLOAD_TYPE_RELATION    = 3

# ---------------------------------------------------------------------------
# Tool schemas — embedded verbatim into inference prompts
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "route_to_stm",
        "description": (
            "Route to Short-Term Memory (STM). "
            "Use when the query references recent context, the current "
            "conversation turn, or something that was just said or done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's natural language query."
                },
                "payload": {
                    "type": "string",
                    "description": "Content to store or retrieve from STM."
                }
            },
            "required": ["query", "payload"]
        }
    },
    {
        "name": "route_to_ltm",
        "description": (
            "Route to Long-Term Memory (LTM). "
            "Use when the query states a persistent user preference, a rule "
            "that should always apply, or knowledge to retain across sessions. "
            "This includes preferences for programming languages, tools, frameworks, "
            "editors, or working styles — even if the preference mentions code or "
            "technology names like TypeScript, Python, or Rust. "
            "Do NOT use this for code analysis, syntax fixing, or file traversal tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's natural language query."
                },
                "payload": {
                    "type": "string",
                    "description": "Content to store or retrieve from LTM."
                }
            },
            "required": ["query", "payload"]
        }
    },
    {
        "name": "route_to_tree",
        "description": (
            "Route to Tree Memory. "
            "Use ONLY when the query includes an actual file path (e.g. main.rs, app.py, "
            "src/utils.ts) OR a code block the user wants analyzed or debugged. "
            "The query must reference a concrete file, module, or pasted code — not just "
            "mention programming concepts or describe a problem in plain English. "
            "Do NOT use for conversational questions about code, tools, CLIs, or errors "
            "described in words without an attached file or code snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's natural language query."
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the file or module being analysed."
                }
            },
            "required": ["query", "file_path"]
        }
    },
    {
        "name": "route_to_cloud",
        "description": (
            "Escalate to Cloud / external LLM. "
            "Use when the query requires general world knowledge, factual information "
            "(geography, history, science, capitals of countries, definitions), "
            "current events, or any information that was NOT explicitly stored by the "
            "user in a prior session. If the user never told you to remember something, "
            "do NOT route to STM or LTM — route here instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's natural language query."
                }
            },
            "required": ["query"]
        }
    }
]

# Convenience set of valid route names for fast membership checks
VALID_ROUTES: set[str] = {s["name"] for s in TOOL_SCHEMAS}

# FunctionGemma requires tools wrapped in {"type": "function", "function": {...}}
# and passed via apply_chat_template(tools=FG_TOOL_SCHEMAS) — NOT embedded in the message body.
FG_TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": schema}
    for schema in TOOL_SCHEMAS
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _key_from_query(query: str) -> str:
    """Deterministic short key from a query string."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]

def _base_header(payload_type: int) -> dict:
    ts = _now_ms()
    return {
        "payload_type": payload_type,
        "version": 1,
        "created_ts":     {"Timestamp": ts},
        "last_access_ts": {"Timestamp": ts},
        "access_count": 1
    }

def _write(key: str, payload_type: int, data: dict[str, Any]) -> dict:
    """Send a WRITE request to the Rust Kernel."""
    body = {
        "op": "insert",
        "key": key,
        "payload": {
            "header": _base_header(payload_type),
            "data": data
        }
    }
    try:
        r = httpx.post(f"{KERNEL_URL}/write", json=body, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"error": str(e), "key": key}

# ---------------------------------------------------------------------------
# Dispatch functions — one per route
# ---------------------------------------------------------------------------

def dispatch_stm(query: str, payload: str) -> dict:
    """Write a TokenStats entry to STM."""
    key = _key_from_query(query)
    data = {
        "TokenStats": {
            "canonical_form": payload,
            "frequency": 1,
            "decay_score": 1.0,
            "preferred_tokenizer_origin": "router"
        }
    }
    result = _write(key, PAYLOAD_TYPE_TOKEN_STATS, data)
    return {"route": "stm", "key": key, "kernel": result}


def dispatch_ltm(query: str, payload: str) -> dict:
    """Write a TokenStats entry to LTM (persisted across sessions)."""
    key = _key_from_query(query)
    data = {
        "TokenStats": {
            "canonical_form": payload,
            "frequency": 1,
            "decay_score": 1.0,
            "preferred_tokenizer_origin": "router"
        }
    }
    result = _write(key, PAYLOAD_TYPE_TOKEN_STATS, data)
    return {"route": "ltm", "key": key, "kernel": result}


def dispatch_tree(query: str, file_path: str) -> dict:
    """Write a CodeNode entry to Tree Memory."""
    key = _key_from_query(query)
    data = {
        "CodeNode": {
            "file_path": file_path,
            "node_type": "file",
            "content_hash": _key_from_query(file_path),
            "children": []
        }
    }
    result = _write(key, PAYLOAD_TYPE_CODE_NODE, data)
    return {"route": "tree", "key": key, "file_path": file_path, "kernel": result}


def dispatch_cloud(query: str) -> dict:
    """Cloud escalation — no kernel write, return sentinel."""
    return {
        "route": "cloud",
        "key": None,
        "kernel": None,
        "escalate": True,
        "query": query
    }


# ---------------------------------------------------------------------------
# Tool map — name → dispatch callable
# ---------------------------------------------------------------------------

TOOL_MAP: dict[str, callable] = {
    "route_to_stm":   dispatch_stm,
    "route_to_ltm":   dispatch_ltm,
    "route_to_tree":  dispatch_tree,
    "route_to_cloud": dispatch_cloud,
}
