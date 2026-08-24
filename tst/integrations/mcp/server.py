"""Minimal MCP JSON-RPC stdio server for environments without an MCP SDK."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from tst.service.service import TSTService

from .tools import call_tool, tool_definitions


def serve(service: TSTService | None = None, *, project: str | Path | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    owns_service = service is None
    current = service or TSTService(project)
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    try:
        for line in input_stream:
            if not line.strip():
                continue
            request_id: Any = None
            try:
                request = json.loads(line)
                request_id = request.get("id")
                response = _dispatch(current, request)
                if response is not None:
                    _write(output_stream, {"jsonrpc": "2.0", "id": request_id, "result": response})
            except Exception as exc:
                if request_id is not None:
                    _write(
                        output_stream,
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32000, "message": str(exc)},
                        },
                    )
    finally:
        if owns_service:
            current.close()


def _dispatch(service: TSTService, request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    params = request.get("params") or {}
    if method == "notifications/initialized" or method == "notifications/cancelled":
        return None
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tst", "version": "0.3.0"},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": tool_definitions()}
    if method == "tools/call":
        name = str(params.get("name", ""))
        result = call_tool(service, name, params.get("arguments") or {})
        return {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            "structuredContent": result,
            "isError": False,
        }
    raise ValueError(f"unknown MCP method: {method}")


def _write(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, separators=(",", ":"), default=str) + "\n")
    stream.flush()
