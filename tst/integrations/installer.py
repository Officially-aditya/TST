"""Safe, explicit project-local skill file installation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .skills import SKILLS


def install_skills(root: str | Path, base_directory: str, *, force: bool = False) -> dict[str, str]:
    project_root = Path(root).expanduser().resolve()
    written: dict[str, str] = {}
    for name, content in SKILLS.items():
        target = project_root / base_directory / name / "SKILL.md"
        _assert_under(target, project_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            written[str(target)] = "exists"
            continue
        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        temporary.write_text(content, encoding="utf-8")
        try:
            temporary.chmod(0o600)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        written[str(target)] = "written"
    return written


def _assert_under(path: Path, root: Path) -> None:
    try:
        path.parent.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("integration target escaped project root") from exc


def write_json_file(root: str | Path, relative: str, value: object, *, force: bool = False) -> str:
    project_root = Path(root).expanduser().resolve()
    target = project_root / relative
    _assert_under(target, project_root)
    if target.exists() and not force:
        return "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        temporary.chmod(0o600)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "written"


def write_text_file(root: str | Path, relative: str, value: str, *, force: bool = False) -> str:
    project_root = Path(root).expanduser().resolve()
    target = project_root / relative
    _assert_under(target, project_root)
    if target.exists() and not force:
        return "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    try:
        temporary.chmod(0o600)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "written"
