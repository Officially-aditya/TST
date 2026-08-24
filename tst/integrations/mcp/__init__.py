"""Small stdio MCP-compatible adapter over TSTService."""

from .server import serve
from .tools import call_tool, tool_definitions

__all__ = ["call_tool", "serve", "tool_definitions"]
