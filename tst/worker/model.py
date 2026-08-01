"""Dependency-light deterministic code-review worker orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tst.analysis.graph_builder import CodeGraph

from .context import ReviewContextBuilder
from .prompts import review_messages
from .schemas import CodeReviewOutput, ReviewValidationResult
from .validator import ReviewValidator

GenerationOutput = str | bytes | Mapping[str, Any]
GenerationFunction = Callable[..., GenerationOutput]


@dataclass(slots=True)
class WorkerMetrics:
    review_requests: int = 0
    generation_failures: int = 0
    parse_failures: int = 0
    accepted_issues: int = 0
    rejected_issues: int = 0
    validation_failures: int = 0
    low_confidence_issues: int = 0


class ReviewWorker:
    def __init__(
        self,
        project_root: str | Path,
        generate: GenerationFunction,
        *,
        graph: CodeGraph | None = None,
        confidence_threshold: float = 0.55,
        debug_raw_output: bool = False,
        max_output_bytes: int = 2 * 1024 * 1024,
        max_prompt_characters: int = 80_000,
        max_new_tokens: int = 2_048,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.generate = generate
        self.max_new_tokens = max_new_tokens
        self.context_builder = ReviewContextBuilder(
            project_root,
            graph=graph,
            max_prompt_characters=max_prompt_characters,
        )
        self.validator = ReviewValidator(
            project_root,
            graph=graph,
            confidence_threshold=confidence_threshold,
            max_raw_output_bytes=max_output_bytes,
            debug_raw_output=debug_raw_output,
            require_content_hash=True,
        )
        self.metrics = WorkerMetrics()

    def review(
        self,
        task: str,
        target_file: str | Path,
        *,
        related_files: tuple[str | Path, ...] = (),
        conventions: tuple[str, ...] = (),
    ) -> ReviewValidationResult:
        self.metrics.review_requests += 1
        context = self.context_builder.build(
            task,
            target_file,
            related_files=related_files,
            conventions=conventions,
        )
        try:
            raw = self.generate(
                messages=review_messages(context),
                response_schema=CodeReviewOutput.model_json_schema(),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
            )
        except Exception as exc:
            self.metrics.generation_failures += 1
            return ReviewValidationResult(
                parse_error=f"review generation failed ({type(exc).__name__})"
            )

        result = self.validator.validate(raw)
        result.accepted.sort(key=_issue_sort_key)
        result.flagged_low_confidence.sort(key=_issue_sort_key)
        result.rejected.sort(key=lambda item: item.index if item.index is not None else -1)
        self.metrics.parse_failures += int(result.parse_error is not None)
        self.metrics.accepted_issues += len(result.accepted)
        self.metrics.rejected_issues += len(result.rejected)
        self.metrics.validation_failures += len(result.rejected)
        self.metrics.low_confidence_issues += len(result.flagged_low_confidence)
        return result

    def status(self) -> dict[str, int]:
        return asdict(self.metrics)


def _issue_sort_key(issue: Any) -> tuple[str, int, int, str]:
    return (issue.file_path, issue.line_start, issue.line_end, issue.title.casefold())


__all__ = ["GenerationFunction", "ReviewWorker", "WorkerMetrics"]
