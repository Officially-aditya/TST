"""Ground structured review findings in files, line ranges, and symbols."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tst.analysis.graph_builder import CodeGraph
from tst.analysis.scanner import UnsafePathError, resolve_project_path

from .schemas import (
    CodeIssue,
    CodeReviewOutput,
    RejectedFinding,
    ReviewValidationResult,
)


class ReviewValidator:
    def __init__(
        self,
        project_root: str | Path,
        *,
        graph: CodeGraph | None = None,
        confidence_threshold: float = 0.55,
        max_raw_output_bytes: int = 2 * 1024 * 1024,
        debug_raw_output: bool = False,
        require_content_hash: bool = False,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve(strict=True)
        if not self.project_root.is_dir():
            raise ValueError("review project root must be a directory")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be between zero and one")
        if max_raw_output_bytes <= 0:
            raise ValueError("raw output limit must be positive")
        self.graph = graph
        self.confidence_threshold = confidence_threshold
        self.max_raw_output_bytes = max_raw_output_bytes
        self.debug_raw_output = debug_raw_output
        self.require_content_hash = require_content_hash

    def validate(self, raw_output: str | bytes | Mapping[str, Any]) -> ReviewValidationResult:
        if isinstance(raw_output, bytes):
            if len(raw_output) > self.max_raw_output_bytes:
                return self._parse_failure("worker output exceeds the configured size limit", "")
            try:
                raw = raw_output.decode("utf-8")
            except UnicodeDecodeError as exc:
                return self._parse_failure(f"worker output is not UTF-8: {exc}", "")
            candidate: Any = raw
        elif isinstance(raw_output, str):
            raw = raw_output
            if len(raw.encode("utf-8")) > self.max_raw_output_bytes:
                return self._parse_failure("worker output exceeds the configured size limit", raw)
            candidate = raw
        else:
            try:
                raw = json.dumps(raw_output, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                return self._parse_failure(
                    f"worker output cannot be serialized: {exc}", repr(raw_output)
                )
            if len(raw.encode("utf-8")) > self.max_raw_output_bytes:
                return self._parse_failure("worker output exceeds the configured size limit", raw)
            candidate = raw_output

        try:
            if isinstance(candidate, str):
                decoded = json.loads(candidate)
            else:
                decoded = candidate
            review = CodeReviewOutput.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            return self._parse_failure(
                f"structured review parse failed ({type(exc).__name__})", raw
            )

        result = ReviewValidationResult(raw_output=self._debug_raw(raw))
        seen: set[tuple[str, int, int, str, str]] = set()
        for index, issue in enumerate(review.issues):
            rejection = self._validate_issue(index, issue, seen)
            if rejection is not None:
                result.rejected.append(rejection)
            elif issue.confidence < self.confidence_threshold:
                result.flagged_low_confidence.append(issue)
            else:
                result.accepted.append(issue)
        return result

    def _parse_failure(self, reason: str, raw: str) -> ReviewValidationResult:
        return ReviewValidationResult(parse_error=reason, raw_output=self._debug_raw(raw))

    def _debug_raw(self, raw: str) -> str | None:
        if not self.debug_raw_output:
            return None
        encoded = raw.encode("utf-8")[: self.max_raw_output_bytes]
        return encoded.decode("utf-8", errors="replace")

    def _validate_issue(
        self,
        index: int,
        issue: CodeIssue,
        seen: set[tuple[str, int, int, str, str]],
    ) -> RejectedFinding | None:
        try:
            path = resolve_project_path(self.project_root, issue.file_path)
        except (UnsafePathError, OSError) as exc:
            return RejectedFinding(index=index, reason=f"invalid source file: {exc}", issue=issue)
        if not path.is_file():
            return RejectedFinding(
                index=index, reason="finding does not refer to a file", issue=issue
            )
        try:
            contents = path.read_bytes()
            text = contents.decode("utf-8")
            line_count = len(text.splitlines())
        except (OSError, UnicodeError) as exc:
            return RejectedFinding(
                index=index, reason=f"source file is unreadable: {exc}", issue=issue
            )
        if issue.line_end > line_count:
            return RejectedFinding(
                index=index,
                reason=f"line range ends at {issue.line_end}, but the file has {line_count} lines",
                issue=issue,
            )
        content_hash = hashlib.sha256(contents).hexdigest()
        if self.require_content_hash and issue.content_hash is None:
            return RejectedFinding(
                index=index,
                reason="finding does not include the required source content hash",
                issue=issue,
            )
        if issue.content_hash is not None and issue.content_hash != content_hash:
            return RejectedFinding(
                index=index,
                reason="finding source content hash is stale",
                issue=issue,
            )

        if self.graph is not None and issue.related_symbols:
            missing = [
                symbol
                for symbol in issue.related_symbols
                if not self.graph.find(symbol, file_path=issue.file_path)
                and not self.graph.find(symbol.rsplit(".", 1)[-1], file_path=issue.file_path)
            ]
            if missing:
                return RejectedFinding(
                    index=index,
                    reason=f"related symbols are not present in the indexed file: {', '.join(missing)}",
                    issue=issue,
                )

        dedupe_key = (
            issue.file_path.casefold(),
            issue.line_start,
            issue.line_end,
            issue.category.value,
            " ".join(issue.title.casefold().split()),
        )
        if dedupe_key in seen:
            return RejectedFinding(index=index, reason="duplicate finding", issue=issue)
        seen.add(dedupe_key)
        return None


# Re-exporting the result beside the validator keeps the public import concise.
__all__ = ["ReviewValidationResult", "ReviewValidator"]
