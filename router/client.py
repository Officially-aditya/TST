"""
router/client.py

Test client for the TST Memory Router server (router/server.py).

Sends 4 canonical test queries — one per memory path — to POST /route,
prints per-query telemetry, and saves all responses to router/responses.json.

Usage:
  # Start the router server first:
  python -m router.server

  # Then in a separate terminal:
  python -m router.client
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL    = "http://127.0.0.1:8003"
OUTPUT_FILE = Path(__file__).parent / "responses.json"
TIMEOUT     = 120.0  # seconds — model cold-start can be slow

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("router.client")

# ---------------------------------------------------------------------------
# Canonical test queries (one per route)
# ---------------------------------------------------------------------------

TEST_CASES: list[dict] = [
    {
        "label":   "STM — recent context",
        "query":   "What did we just discuss?",
        "payload": "Recent conversation turn about memory routing.",
    },
    {
        "label":   "LTM — persistent preference",
        "query":   "User always prefers TypeScript over JavaScript.",
        "payload": "TypeScript preference rule — persist across sessions.",
    },
    {
        "label":   "Tree — code analysis",
        "query":   "Fix the syntax error on line 53 of main.rs",
        "payload": "",
    },
    {
        "label":   "Cloud — world knowledge",
        "query":   "What is the capital of France?",
        "payload": "",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_telemetry(t: dict) -> None:
    log.info("  Tier         : %s", t.get("tier"))
    log.info("  Model        : %s", t.get("model"))
    log.info("  Prompt tokens: %d", t.get("prompt_tokens", 0))
    log.info("  Eval tokens  : %d", t.get("eval_tokens", 0))
    log.info("  Wall time    : %.1f ms", t.get("wall_time_ms", 0))
    log.info("  Tokens/sec   : %.1f", t.get("tokens_per_sec", 0))


def send_route(client: httpx.Client, query: str, payload: str) -> dict:
    resp = client.post(
        f"{BASE_URL}/route",
        json={"query": query, "payload": payload},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Liveness check
    try:
        with httpx.Client() as c:
            health = c.get(f"{BASE_URL}/health", timeout=5.0)
        log.info("Server health: %s", health.json())
    except httpx.ConnectError:
        log.error("Cannot reach router server at %s — is it running?", BASE_URL)
        sys.exit(1)

    stored: list[dict] = []

    with httpx.Client() as client:
        for i, tc in enumerate(TEST_CASES, 1):
            log.info("")
            log.info("─── Test %d / %d: %s ───", i, len(TEST_CASES), tc["label"])
            log.info("  Query  : %s", tc["query"])
            log.info("  Payload: %s", tc["payload"] or "(empty)")

            result = send_route(client, tc["query"], tc["payload"])

            log.info("  → Tool called : %s", result.get("tool_called"))
            log.info("  → Args        : %s", json.dumps(result.get("args", {})))
            log.info("  → Route result: %s", result.get("result", {}).get("route"))
            log.info("  → Key         : %s", result.get("result", {}).get("key"))
            log.info("  → Escalate    : %s", result.get("result", {}).get("escalate"))
            _print_telemetry(result.get("telemetry", {}))

            stored.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "label":     tc["label"],
                "query":     tc["query"],
                "payload":   tc["payload"],
                "tool_called": result.get("tool_called"),
                "args":      result.get("args"),
                "result":    result.get("result"),
                "telemetry": result.get("telemetry"),
            })

    OUTPUT_FILE.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    log.info("")
    log.info("Saved %d responses → %s", len(stored), OUTPUT_FILE)

    # Summary
    correct = sum(
        1 for tc, s in zip(TEST_CASES, stored)
        if tc["label"].split(" — ")[0].lower() in s.get("tool_called", "")
    )
    log.info("Routing accuracy: %d / %d", correct, len(TEST_CASES))


if __name__ == "__main__":
    main()
