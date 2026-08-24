"""Persistent registry of repositories known to the local TST installation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock

from .models import Project
from .project import ensure_project


class ProjectRegistry:
    """Store project identities separately from project-local memory snapshots."""

    def __init__(self, home: str | Path | None = None) -> None:
        configured = os.environ.get("TST_HOME")
        self.home = Path(home or configured or (Path.home() / ".tst")).expanduser().resolve()
        self.global_dir = self.home / "global"
        self.projects_path = self.home / "projects.json"
        self._lock = RLock()

    @property
    def global_snapshot(self) -> Path:
        return self.global_dir / "ltm.snapshot"

    def ensure_layout(self) -> None:
        self.global_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Project]:
        with self._lock:
            return sorted(self._read(), key=lambda project: (project.name.casefold(), str(project.root)))

    def projects(self) -> list[Project]:
        """Named alias for callers that prefer the application vocabulary."""

        return self.list()

    def register(self, project: Project) -> Project:
        with self._lock:
            self.ensure_layout()
            projects = self._read()
            replaced = False
            for index, current in enumerate(projects):
                if current.id == project.id or current.root == project.root:
                    projects[index] = project
                    replaced = True
                    break
            if not replaced:
                projects.append(project)
            self._write(projects)
        return project

    def ensure(self, root: str | Path) -> Project:
        project = ensure_project(root)
        return self.register(project)

    def find(self, value: str | Path) -> Project | None:
        text = str(value)
        resolved: Path | None = None
        try:
            resolved = Path(value).expanduser().resolve()
        except (OSError, RuntimeError):
            pass
        with self._lock:
            for project in self._read():
                if project.id == text or project.name == text:
                    return project
                if resolved is not None and project.root == resolved:
                    return project
        return None

    def resolve(self, value: str | Path) -> Project:
        found = self.find(value)
        if found is None:
            raise KeyError(f"project is not registered: {value}")
        return found

    def remove(self, value: str | Path) -> bool:
        with self._lock:
            found = self.find(value)
            if found is None:
                return False
            projects = [project for project in self._read() if project.id != found.id]
            self._write(projects)
            return True

    def _read(self) -> list[Project]:
        if not self.projects_path.is_file():
            return []
        try:
            document = json.loads(self.projects_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid project registry: {self.projects_path}") from exc
        raw_projects = document.get("projects", []) if isinstance(document, dict) else document
        if not isinstance(raw_projects, list):
            raise ValueError("project registry must contain a list of projects")
        projects: list[Project] = []
        for raw in raw_projects:
            if isinstance(raw, dict):
                projects.append(Project.from_dict(raw))
        return projects

    def _write(self, projects: list[Project]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        temporary = self.projects_path.with_name(f".{self.projects_path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps([project.to_dict() for project in projects], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            temporary.chmod(0o600)
            os.replace(temporary, self.projects_path)
        finally:
            if temporary.exists():
                temporary.unlink()
