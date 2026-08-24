"""Safe, explicit project-local skill file installation."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

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


def merge_json_file(
    root: str | Path,
    relative: str,
    value: dict[str, Any],
    *,
    force: bool = False,
    append_lists: set[tuple[str, ...]] | None = None,
) -> str:
    """Merge project integration settings without replacing unrelated config."""

    project_root = Path(root).expanduser().resolve()
    target = project_root / relative
    _assert_under(target, project_root)
    if target.exists():
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "invalid"
        if not isinstance(current, dict):
            return "invalid"
    else:
        current = {}

    candidate = deepcopy(current)
    changed, conflict = _merge_values(candidate, value, force=force, append_lists=append_lists or set())
    if conflict:
        return "conflict"
    if not changed:
        return "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def merge_text_block(
    root: str | Path,
    relative: str,
    value: str,
    marker: str,
    *,
    force: bool = False,
) -> str:
    """Append or replace a marked configuration block without clobbering peers."""

    project_root = Path(root).expanduser().resolve()
    target = project_root / relative
    _assert_under(target, project_root)
    begin = f"# {marker} BEGIN"
    end = f"# {marker} END"
    block = f"{begin}\n{value.rstrip()}\n{end}\n"
    if not target.exists():
        return _write_text_document(target, block)
    try:
        current = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "invalid"
    start = current.find(begin)
    finish = current.find(end, start + len(begin)) if start >= 0 else -1
    if start >= 0 and finish >= 0:
        finish += len(end)
        existing = current[start:finish]
        if existing == block.rstrip("\n"):
            return "exists"
        if not force:
            return "conflict"
        updated = current[:start] + block.rstrip("\n") + current[finish:]
        return _write_text_document(target, updated)
    if begin in current or end in current:
        return "invalid"
    if "[mcp_servers.tst]" in current:
        if current.strip() == value.strip():
            return "exists"
        if force:
            lines = current.splitlines(keepends=True)
            start_index = next(
                index for index, line in enumerate(lines) if line.strip() == "[mcp_servers.tst]"
            )
            end_index = next(
                (
                    index
                    for index in range(start_index + 1, len(lines))
                    if lines[index].lstrip().startswith("[")
                ),
                len(lines),
            )
            updated = "".join(lines[:start_index]) + block + "".join(lines[end_index:])
            return _write_text_document(target, updated)
        return "conflict"
    separator = "" if not current or current.endswith("\n") else "\n"
    return _write_text_document(target, current + separator + block)


def remove_text_block(
    root: str | Path,
    relative: str,
    value: str,
    marker: str,
    *,
    force: bool = False,
) -> str:
    """Remove one marked block and retain the rest of the configuration file."""

    project_root = Path(root).expanduser().resolve()
    target = project_root / relative
    _assert_under(target, project_root)
    if not target.exists():
        return "absent"
    try:
        current = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "invalid"
    begin = f"# {marker} BEGIN"
    end = f"# {marker} END"
    start = current.find(begin)
    finish = current.find(end, start + len(begin)) if start >= 0 else -1
    if start < 0 or finish < 0:
        return remove_text_file(root, relative, value, force=force)
    finish += len(end)
    block = current[start:finish]
    expected = f"{begin}\n{value.rstrip()}\n{end}"
    if not force and block != expected:
        return "modified"
    updated = current[:start] + current[finish:]
    updated = updated.lstrip("\n") if not updated.strip() else updated.strip("\n") + "\n"
    if not updated:
        target.unlink()
        _remove_empty_parents(target.parent, project_root)
        return "removed"
    _write_text_document(target, updated)
    return "removed"


def remove_text_file(
    root: str | Path,
    relative: str,
    expected: str,
    *,
    force: bool = False,
) -> str:
    """Remove a generated file without deleting user edits by default."""

    project_root = Path(root).expanduser().resolve()
    target = project_root / relative
    _assert_under(target, project_root)
    if not target.exists():
        return "absent"
    try:
        current = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "invalid"
    if not force and current != expected:
        return "modified"
    target.unlink()
    _remove_empty_parents(target.parent, project_root)
    return "removed"


def _write_text_document(target: Path, value: str) -> str:
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


def remove_json_key(
    root: str | Path,
    relative: str,
    path: tuple[str, ...],
    expected: object,
    *,
    force: bool = False,
) -> str:
    """Remove one generated JSON setting while preserving sibling settings."""

    project_root = Path(root).expanduser().resolve()
    target = project_root / relative
    _assert_under(target, project_root)
    if not target.exists():
        return "absent"
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid"
    if not isinstance(document, dict):
        return "invalid"
    parent: dict[str, Any] = document
    for key in path[:-1]:
        value = parent.get(key)
        if not isinstance(value, dict):
            return "absent"
        parent = value
    key = path[-1]
    if key not in parent:
        return "absent"
    if not force and parent[key] != expected:
        return "modified"
    del parent[key]
    _prune_empty_dicts(document, path[:-1])
    _write_json_document(target, document)
    return "removed"


def remove_json_list_item(
    root: str | Path,
    relative: str,
    path: tuple[str, ...],
    expected: object,
    *,
    force: bool = False,
) -> str:
    """Remove one generated list entry while preserving user entries."""

    project_root = Path(root).expanduser().resolve()
    target = project_root / relative
    _assert_under(target, project_root)
    if not target.exists():
        return "absent"
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid"
    if not isinstance(document, dict):
        return "invalid"
    value: object = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return "absent"
        value = value[key]
    if not isinstance(value, list):
        return "invalid"
    matches = [index for index, item in enumerate(value) if item == expected]
    if not matches:
        return "absent"
    if len(matches) > 1 and not force:
        return "modified"
    del value[matches[-1]]
    _prune_empty_dicts(document, path)
    _write_json_document(target, document)
    return "removed"


def _write_json_document(target: Path, document: dict[str, Any]) -> None:
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        temporary.chmod(0o600)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prune_empty_dicts(document: dict[str, Any], path: tuple[str, ...]) -> None:
    for index in range(len(path), 0, -1):
        parent: object = document
        for key in path[: index - 1]:
            if not isinstance(parent, dict):
                break
            parent = parent.get(key)
        if not isinstance(parent, dict):
            continue
        key = path[index - 1]
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            del parent[key]


def _remove_empty_parents(directory: Path, root: Path) -> None:
    current = directory
    while current != root:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _merge_values(
    current: dict[str, Any],
    patch: dict[str, Any],
    *,
    force: bool,
    append_lists: set[tuple[str, ...]],
    path: tuple[str, ...] = (),
) -> tuple[bool, bool]:
    changed = False
    for key, desired in patch.items():
        existing = current.get(key)
        if key not in current:
            current[key] = deepcopy(desired)
            changed = True
            continue
        if isinstance(existing, dict) and isinstance(desired, dict):
            nested_changed, conflict = _merge_values(
                existing,
                desired,
                force=force,
                append_lists=append_lists,
                path=(*path, key),
            )
            if conflict:
                return changed, True
            changed = changed or nested_changed
            continue
        if existing == desired:
            continue
        if path + (key,) in append_lists and isinstance(existing, list) and isinstance(desired, list):
            for item in desired:
                if item not in existing:
                    existing.append(deepcopy(item))
                    changed = True
            continue
        if not force:
            return changed, True
        current[key] = deepcopy(desired)
        changed = True
    return changed, False
