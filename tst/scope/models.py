"""Typed values shared by project discovery, storage, and service layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4


class Scope(str, Enum):
    """The lifetime and storage boundary of a memory item."""

    GLOBAL = "global"
    PROJECT = "project"
    SESSION = "session"

    @classmethod
    def coerce(cls, value: Scope | str) -> Scope:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unknown memory scope: {value!r}") from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class Project:
    """A registered repository identity.

    The UUID is stable for a repository once ``.tst/project.json`` exists;
    ``root`` is always absolute so it can safely be passed to a child process.
    """

    id: str
    name: str
    root: Path
    created_at: str

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not self.id.strip():
            raise ValueError("project id cannot be empty")
        if not self.name.strip():
            raise ValueError("project name cannot be empty")
        object.__setattr__(self, "root", root)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        name: str | None = None,
        project_id: str | None = None,
        created_at: str | None = None,
    ) -> Project:
        resolved = Path(root).expanduser().resolve()
        return cls(
            id=project_id or str(uuid4()),
            name=name or resolved.name or "project",
            root=resolved,
            created_at=created_at or utc_now(),
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Project:
        root = value.get("root")
        if not isinstance(root, str):
            raise ValueError("project metadata requires a root")
        return cls(
            id=str(value.get("id", "")),
            name=str(value.get("name", "")),
            root=Path(root),
            created_at=str(value.get("created_at", "")) or utc_now(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "root": str(self.root),
            "created_at": self.created_at,
        }

    @property
    def metadata_path(self) -> Path:
        return self.root / ".tst" / "project.json"

    @property
    def snapshot_path(self) -> Path:
        return self.root / ".tst" / "ltm.snapshot"
