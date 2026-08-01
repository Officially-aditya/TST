from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tst.worker.context import ReviewContextBuilder
from tst.worker.model import ReviewWorker
from tst.worker.validator import ReviewValidator


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _issue(content_hash: str | None, **updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "file_path": "app.py",
        "content_hash": content_hash,
        "line_start": 1,
        "line_end": 2,
        "severity": "high",
        "category": "logic",
        "title": "Incorrect constant",
        "explanation": "run returns zero when callers require one.",
        "suggested_fix": "Return one.",
        "confidence": 0.9,
        "related_symbols": [],
    }
    value.update(updates)
    return value


def test_worker_uses_deterministic_generation_and_tracks_metrics(tmp_path: Path) -> None:
    source = "def run():\n    return 0\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    captured: dict[str, Any] = {}

    def generate(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "issues": [
                _issue(_hash(source), title="Zulu"),
                _issue(_hash(source), title="Alpha", line_start=2, line_end=2),
            ]
        }

    worker = ReviewWorker(tmp_path, generate)
    result = worker.review("Find correctness defects", "app.py")

    assert captured["do_sample"] is False
    assert captured["temperature"] == 0.0
    assert captured["max_new_tokens"] == 2_048
    assert captured["response_schema"]["type"] == "object"
    assert [issue.title for issue in result.accepted] == ["Zulu", "Alpha"]
    assert result.raw_output is None
    assert worker.status() == {
        "review_requests": 1,
        "generation_failures": 0,
        "parse_failures": 0,
        "accepted_issues": 2,
        "rejected_issues": 0,
        "validation_failures": 0,
        "low_confidence_issues": 0,
    }


def test_context_has_named_sections_hashes_and_no_absolute_root(tmp_path: Path) -> None:
    source = "def run():\n    return helper()\n"
    related = "def helper():\n    return 1\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    (tmp_path / "helper.py").write_text(related, encoding="utf-8")

    context = ReviewContextBuilder(tmp_path).build(
        "Review run",
        "app.py",
        related_files=("helper.py",),
        conventions=("Use explicit return types", "Use explicit return types"),
    )

    for section in (
        "Task",
        "Target code",
        "Relevant symbol graph",
        "Related source excerpts",
        "Retrieved project conventions",
        "Output schema",
    ):
        assert f"## {section}" in context.prompt
    assert context.content_hash == _hash(source)
    assert str(tmp_path) not in context.prompt
    assert "app.py" in context.prompt and "helper.py" in context.prompt
    assert context.prompt.count("Use explicit return types") == 1


def test_worker_rejects_missing_and_stale_source_hashes(tmp_path: Path) -> None:
    source = "def run():\n    return 0\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")

    worker = ReviewWorker(
        tmp_path,
        lambda **_: {
            "issues": [
                _issue(None, title="Missing hash"),
                _issue("0" * 64, title="Stale hash"),
            ]
        },
    )
    result = worker.review("Review", "app.py")
    assert result.accepted == []
    assert {finding.reason for finding in result.rejected} == {
        "finding does not include the required source content hash",
        "finding source content hash is stale",
    }
    assert worker.status()["validation_failures"] == 2


def test_generation_failures_and_malformed_output_are_safe(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("pass\n", encoding="utf-8")

    failed = ReviewWorker(
        tmp_path,
        lambda **_: (_ for _ in ()).throw(RuntimeError("secret failure detail")),
    )
    failure = failed.review("Review", "app.py")
    assert failure.accepted == []
    assert failure.raw_output is None
    assert failure.parse_error == "review generation failed (RuntimeError)"
    assert failed.status()["generation_failures"] == 1

    malformed = ReviewWorker(tmp_path, lambda **_: "not JSON")
    parse_failure = malformed.review("Review", "app.py")
    assert parse_failure.accepted == []
    assert parse_failure.raw_output is None
    assert parse_failure.parse_error == "structured review parse failed (JSONDecodeError)"
    assert malformed.status()["parse_failures"] == 1


def test_raw_output_is_explicit_and_bounded(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("pass\n", encoding="utf-8")
    raw = json.dumps({"issues": []})
    private = ReviewValidator(tmp_path).validate(raw)
    assert private.raw_output is None

    debug = ReviewValidator(tmp_path, debug_raw_output=True).validate(raw)
    assert debug.raw_output == raw

    oversized = ReviewValidator(
        tmp_path,
        debug_raw_output=True,
        max_raw_output_bytes=16,
    ).validate("x" * 100)
    assert oversized.parse_error == "worker output exceeds the configured size limit"
    assert oversized.raw_output is not None
    assert len(oversized.raw_output.encode("utf-8")) <= 16


def test_worker_enforces_output_size_and_accepts_mapping_output(tmp_path: Path) -> None:
    source = "def run():\n    return 0\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")

    mapping_worker = ReviewWorker(
        tmp_path,
        lambda **_: {"issues": [_issue(_hash(source))]},
    )
    assert len(mapping_worker.review("Review", "app.py").accepted) == 1

    oversized = ReviewWorker(
        tmp_path,
        lambda **_: "x" * 1_000,
        max_output_bytes=64,
        debug_raw_output=True,
    ).review("Review", "app.py")
    assert oversized.parse_error == "worker output exceeds the configured size limit"
    assert oversized.raw_output is not None
    assert len(oversized.raw_output.encode("utf-8")) <= 64
