"""Stable prompts for structured, source-grounded code review."""

from __future__ import annotations

from .context import ReviewContext

REVIEW_SYSTEM_PROMPT = """You are a deterministic code-review worker.
Return exactly one JSON object matching the supplied Output schema and no prose.
Report only concrete defects supported by the supplied source. Use root-relative
paths and exact inclusive line ranges. Copy the target's sha256 into content_hash
for every target-file issue. Put every cited symbol in related_symbols. If there
are no supported defects, return {"issues": []}. Never propose or execute edits.
"""


def review_messages(context: ReviewContext) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": context.prompt},
    ]


__all__ = ["REVIEW_SYSTEM_PROMPT", "review_messages"]
