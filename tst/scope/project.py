"""Safe project-root discovery and project metadata persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import Project


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest repository marker without executing repository code."""

    candidate = Path(start or os.getcwd()).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".tst" / "project.json").is_file() or (directory / ".git").exists():
            return directory
    return candidate


def load_project(root: str | Path) -> Project | None:
    path = Path(root).expanduser().resolve() / ".tst" / "project.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid project metadata: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"project metadata must be an object: {path}")
    project = Project.from_dict(document)
    if project.root != path.parent.parent:
        raise ValueError("project metadata root does not match its location")
    return project


def ensure_project(
    root: str | Path,
    *,
    name: str | None = None,
    project_id: str | None = None,
) -> Project:
    """Load project identity or create it atomically on first use."""

    resolved = Path(root).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    existing = load_project(resolved)
    if existing is not None:
        return existing
    project = Project.create(resolved, name=name, project_id=project_id)
    metadata_dir = resolved / ".tst"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_json_private(project.metadata_path, project.to_dict())
    return project


def _write_json_private(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
