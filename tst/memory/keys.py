"""Readable, validated hierarchical keys for stable memory identity."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote, unquote

_RESERVED_PREFIXES = {"_", "internal", "kernel", "protocol", "system"}
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._~%+-]+$")


class InvalidCanonicalKey(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CanonicalKeyBuilder:
    user_id: str = "default"
    session_id: str = "default"
    project_id: str = "default"

    @staticmethod
    def escape(segment: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(segment)).strip().lower()
        normalized = re.sub(r"\s+", "_", normalized)
        if not normalized:
            raise InvalidCanonicalKey("key segments cannot be empty")
        if normalized in {".", ".."}:
            raise InvalidCanonicalKey("dot path segments are not allowed")
        escaped = quote(normalized, safe="-._~+")
        if not _SAFE_SEGMENT.fullmatch(escaped):
            raise InvalidCanonicalKey(f"invalid key segment: {segment!r}")
        return escaped

    @classmethod
    def build(cls, *segments: str) -> str:
        if len(segments) < 2:
            raise InvalidCanonicalKey("canonical keys require at least two segments")
        escaped = [cls.escape(segment) for segment in segments]
        if unquote(escaped[0]).lower() in _RESERVED_PREFIXES:
            raise InvalidCanonicalKey(f"reserved key prefix: {segments[0]!r}")
        key = ":".join(escaped)
        cls.validate(key)
        return key

    @classmethod
    def validate(cls, key: str) -> str:
        if not isinstance(key, str) or len(key) > 512:
            raise InvalidCanonicalKey("key must be a string no longer than 512 characters")
        parts = key.split(":")
        if len(parts) < 2 or any(not part for part in parts):
            raise InvalidCanonicalKey("key must contain non-empty colon-separated segments")
        if unquote(parts[0]).lower() in _RESERVED_PREFIXES:
            raise InvalidCanonicalKey(f"reserved key prefix: {parts[0]!r}")
        if any(not _SAFE_SEGMENT.fullmatch(part) for part in parts):
            raise InvalidCanonicalKey("key contains an invalid or unescaped segment")
        for part in parts:
            if re.search(r"%(?![0-9A-F]{2})", part):
                raise InvalidCanonicalKey("key contains invalid percent escaping")
            try:
                canonical = cls.escape(unquote(part))
            except (UnicodeError, ValueError) as exc:
                raise InvalidCanonicalKey("key contains an invalid encoded segment") from exc
            if canonical != part:
                raise InvalidCanonicalKey("key is not in canonical normalized form")
        return key

    def user_preference(self, subject: str) -> str:
        return self.build("user", self.user_id, "preference", subject)

    def user_fact(self, subject: str) -> str:
        return self.build("user", self.user_id, "fact", subject)

    def session_context(self, subject: str) -> str:
        return self.build("session", self.session_id, "context", subject)

    def session_turn(self, turn: int) -> str:
        if turn < 0:
            raise InvalidCanonicalKey("turn cannot be negative")
        return self.build("session", self.session_id, "turn", str(turn))

    def project_file(self, file_path: str) -> str:
        normalized = str(PurePosixPath(file_path.replace("\\", "/")))
        if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
            raise InvalidCanonicalKey("project file key must use a relative path")
        return self.build("project", self.project_id, "file", normalized)

    def project_symbol(self, qualified_name: str) -> str:
        return self.build("project", self.project_id, "symbol", qualified_name)

    def for_subject(self, layer: str, subject: str, *, memory_type: str = "preference") -> str:
        if layer == "stm":
            return self.session_context(subject)
        if layer == "ltm" and memory_type == "preference":
            return self.user_preference(subject)
        if layer == "ltm":
            return self.user_fact(subject)
        raise InvalidCanonicalKey(f"cannot build a memory key for layer {layer!r}")
