"""Strict schemas used by the deterministic code-review worker."""

from __future__ import annotations

import math
import re
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def normalize_project_file_path(value: Any) -> str:
    """Return a canonical root-relative POSIX path or raise ``ValueError``."""

    if not isinstance(value, str) or value != value.strip():
        raise ValueError("file_path must be a non-empty string without outer whitespace")
    if "\\" in value:
        raise ValueError("file_path must use POSIX separators")
    if "\x00" in value or value.startswith("/") or re.match(r"^[A-Za-z]:/", value):
        raise ValueError("file_path must be a root-relative project path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("file_path cannot contain empty or traversal components")
    return PurePosixPath(value).as_posix()


def validate_content_hash(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
    return value


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, Enum):
    SYNTAX = "syntax"
    LOGIC = "logic"
    RUNTIME = "runtime"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


class CodeIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    file_path: str = Field(min_length=1, max_length=1_024)
    content_hash: str | None = Field(default=None, min_length=64, max_length=64)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    severity: Severity
    category: IssueCategory
    title: str = Field(min_length=1, max_length=240)
    explanation: str = Field(min_length=1, max_length=8_000)
    suggested_fix: str = Field(min_length=1, max_length=8_000)
    confidence: float = Field(ge=0.0, le=1.0)
    related_symbols: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_relative_file_path(cls, value: Any) -> str:
        return normalize_project_file_path(value)

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str | None) -> str | None:
        return validate_content_hash(value) if value is not None else None

    @field_validator("line_start", "line_end", mode="before")
    @classmethod
    def validate_line_number_type(cls, value: Any) -> int:
        if type(value) is not int:
            raise ValueError("line numbers must be integers")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence_type(cls, value: Any) -> float:
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("confidence must be a finite number")
        return float(value)

    @field_validator("related_symbols", mode="before")
    @classmethod
    def validate_symbol_types(cls, value: Any) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("related_symbols must be a list of strings")
        return value

    @field_validator("related_symbols")
    @classmethod
    def normalize_symbols(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for symbol in value:
            item = symbol.strip()
            if not item or len(item) > 500:
                raise ValueError("related symbols must be non-empty and at most 500 characters")
            if item not in normalized:
                normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def validate_line_order(self) -> CodeIssue:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class CodeReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[CodeIssue] = Field(default_factory=list, max_length=1_000)

    @field_validator("issues", mode="before")
    @classmethod
    def validate_issue_collection(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            raise ValueError("issues must be a JSON array")
        return value


class RejectedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int | None = None
    reason: str
    issue: CodeIssue | None = None


class ReviewValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: list[CodeIssue] = Field(default_factory=list)
    flagged_low_confidence: list[CodeIssue] = Field(default_factory=list)
    rejected: list[RejectedFinding] = Field(default_factory=list)
    parse_error: str | None = None
    raw_output: str | None = Field(default=None, repr=False)

    @property
    def safe_to_display(self) -> list[CodeIssue]:
        return self.accepted
