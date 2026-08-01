"""Bounded and path-safe repository source discovery.

The scanner deliberately never executes project code and never follows directory
symlinks.  Its output uses root-relative POSIX paths so graph identities remain
stable across machines.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

DEFAULT_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
}

DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "bower_components",
        "build",
        "coverage",
        "dist",
        "gemma-env",
        "node_modules",
        "site-packages",
        "target",
        "vendor",
        "venv",
    }
)

DEFAULT_SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "*credentials*.json",
    "*credentials*.yaml",
    "*credentials*.yml",
    "*secret*.json",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "id_rsa*",
)


class ScanError(RuntimeError):
    """Base error for repository scanning."""


class UnsafePathError(ScanError):
    """Raised when a requested path escapes its selected project root."""


@dataclass(frozen=True, slots=True)
class ScanConfig:
    languages: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LANGUAGES))
    excluded_directories: frozenset[str] = DEFAULT_EXCLUDED_DIRECTORIES
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS
    max_file_bytes: int = 2 * 1024 * 1024
    max_repository_bytes: int = 64 * 1024 * 1024
    max_files: int = 10_000
    skip_hidden_directories: bool = True
    skip_all_symlinks: bool = True

    def __post_init__(self) -> None:
        if min(self.max_file_bytes, self.max_repository_bytes, self.max_files) <= 0:
            raise ValueError("scanner size and count limits must be positive")


@dataclass(frozen=True, slots=True)
class SourceFile:
    absolute_path: Path
    relative_path: str
    language: str
    size_bytes: int
    content_hash: str

    def read_text(self) -> str:
        """Read source without guessing legacy encodings or hiding decode errors."""

        return self.absolute_path.read_text(encoding="utf-8")


def resolve_project_path(
    root: str | os.PathLike[str],
    requested: str | os.PathLike[str],
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve ``requested`` and prove that it remains inside ``root``.

    Absolute paths are accepted only when they point inside the selected root.
    Resolving both sides also blocks ``..`` traversal and outward-pointing
    symlinks.
    """

    project_root = Path(root).expanduser().resolve(strict=True)
    candidate = Path(requested).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (FileNotFoundError, RuntimeError) as exc:
        raise UnsafePathError(f"cannot resolve project path: {requested}") from exc
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise UnsafePathError(f"path escapes project root: {requested}") from exc
    return resolved


class ProjectScanner:
    def __init__(
        self,
        root: str | os.PathLike[str],
        config: ScanConfig | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ScanError(f"project root is not a directory: {self.root}")
        self.config = config or ScanConfig()
        self.skipped: list[tuple[str, str]] = []

    def _record_skip(self, path: Path, reason: str) -> None:
        try:
            display = path.relative_to(self.root).as_posix()
        except ValueError:
            display = str(path)
        self.skipped.append((display, reason))

    def _is_secret_name(self, name: str) -> bool:
        lowered = name.lower()
        return any(
            fnmatch.fnmatch(lowered, pattern.lower()) for pattern in self.config.secret_patterns
        )

    def scan(self, include_paths: set[str] | None = None) -> list[SourceFile]:
        self.skipped.clear()
        results: list[SourceFile] = []
        total_bytes = 0
        normalized_includes = (
            {PurePosixPath(path).as_posix() for path in include_paths}
            if include_paths is not None
            else None
        )

        for directory, dirnames, filenames in os.walk(self.root, followlinks=False):
            current = Path(directory)
            kept_directories: list[str] = []
            for name in sorted(dirnames):
                candidate = current / name
                if name in self.config.excluded_directories:
                    self._record_skip(candidate, "excluded directory")
                elif self.config.skip_hidden_directories and name.startswith("."):
                    self._record_skip(candidate, "hidden directory")
                elif candidate.is_symlink():
                    self._record_skip(candidate, "directory symlink")
                else:
                    kept_directories.append(name)
            dirnames[:] = kept_directories

            for name in sorted(filenames):
                path = current / name
                relative_candidate = path.relative_to(self.root).as_posix()
                if (
                    normalized_includes is not None
                    and relative_candidate not in normalized_includes
                ):
                    continue
                suffix = path.suffix.lower()
                language = self.config.languages.get(suffix)
                if language is None:
                    continue
                if self._is_secret_name(name):
                    self._record_skip(path, "potential secret")
                    continue
                if path.is_symlink() and self.config.skip_all_symlinks:
                    self._record_skip(path, "file symlink")
                    continue
                try:
                    resolved = resolve_project_path(self.root, path)
                    stat = resolved.stat()
                except (OSError, UnsafePathError) as exc:
                    self._record_skip(path, f"unsafe or unreadable: {type(exc).__name__}")
                    continue
                if not resolved.is_file():
                    continue
                if stat.st_size > self.config.max_file_bytes:
                    self._record_skip(path, "file size limit")
                    continue
                if len(results) >= self.config.max_files:
                    self._record_skip(path, "repository file-count limit")
                    continue
                try:
                    with resolved.open("rb") as stream:
                        content = stream.read(self.config.max_file_bytes + 1)
                    if len(content) > self.config.max_file_bytes:
                        self._record_skip(path, "file size limit")
                        continue
                    if total_bytes + len(content) > self.config.max_repository_bytes:
                        self._record_skip(path, "repository size limit")
                        continue
                    if b"\x00" in content:
                        self._record_skip(path, "binary content")
                        continue
                    content.decode("utf-8")
                    content_hash = hashlib.sha256(content).hexdigest()
                except (OSError, UnicodeDecodeError) as exc:
                    self._record_skip(path, f"unreadable UTF-8 source: {type(exc).__name__}")
                    continue

                relative = resolved.relative_to(self.root).as_posix()
                results.append(
                    SourceFile(
                        absolute_path=resolved,
                        relative_path=relative,
                        language=language,
                        size_bytes=len(content),
                        content_hash=content_hash,
                    )
                )
                total_bytes += len(content)

        return results

    def iter_files(self, include_paths: set[str] | None = None) -> Iterator[SourceFile]:
        yield from self.scan(include_paths)
