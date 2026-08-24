"""Lifecycle manager for the global kernel and one active project kernel."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import RLock

from tst.kernel.client import StdioKernelClient
from tst.kernel.process import KernelProcessConfig

from .models import Project
from .registry import ProjectRegistry

ClientFactory = Callable[[KernelProcessConfig], StdioKernelClient]


class KernelManager:
    """Keep at most one global and one project-scoped child process alive."""

    def __init__(
        self,
        registry: ProjectRegistry | None = None,
        *,
        config: KernelProcessConfig | None = None,
        client_factory: ClientFactory = StdioKernelClient,
    ) -> None:
        self.registry = registry or ProjectRegistry()
        self.config = config or KernelProcessConfig()
        self.client_factory = client_factory
        self._global: StdioKernelClient | None = None
        self._project_client: StdioKernelClient | None = None
        self._project: Project | None = None
        self._lock = RLock()

    @property
    def active_project(self) -> Project | None:
        with self._lock:
            return self._project

    @property
    def global_process(self) -> StdioKernelClient | None:
        return self._global

    @property
    def active_project_process(self) -> StdioKernelClient | None:
        return self._project_client

    def global_client(self) -> StdioKernelClient:
        with self._lock:
            if self._global is not None and self._global.is_running:
                return self._global
            if self._global is not None:
                self._global.close(graceful=False)
            snapshot = self.registry.global_snapshot.expanduser().resolve()
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            client = self.client_factory(self._config_for(snapshot))
            client.start()
            self._global = client
            return client

    def project_client(self, project: Project | str | Path) -> StdioKernelClient:
        selected = self._resolve_project(project)
        with self._lock:
            if (
                self._project_client is not None
                and self._project is not None
                and self._project.root == selected.root
                and self._project_client.is_running
            ):
                return self._project_client
            self._close_project_locked()
            selected.root.joinpath(".tst").mkdir(parents=True, exist_ok=True)
            snapshot = selected.snapshot_path.expanduser().resolve()
            client = self.client_factory(self._config_for(snapshot))
            try:
                client.start()
            except Exception:
                client.close(graceful=False)
                raise
            self._project = selected
            self._project_client = client
            return client

    def restart_project(self, project: Project | str | Path | None = None) -> StdioKernelClient:
        with self._lock:
            selected = self._resolve_project(project or self._project)
            self._close_project_locked()
        return self.project_client(selected)

    def close(self) -> None:
        with self._lock:
            self._close_project_locked()
            if self._global is not None:
                self._global.close(graceful=True)
                self._global = None

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "global": {
                    "running": bool(self._global and self._global.is_running),
                    "pid": self._global.pid if self._global else None,
                    "snapshot": str(self.registry.global_snapshot),
                },
                "project": {
                    "running": bool(self._project_client and self._project_client.is_running),
                    "pid": self._project_client.pid if self._project_client else None,
                    "project": self._project.to_dict() if self._project else None,
                    "snapshot": str(self._project.snapshot_path) if self._project else None,
                },
            }

    def _resolve_project(self, project: Project | str | Path | None) -> Project:
        if isinstance(project, Project):
            return project
        if project is None:
            if self._project is not None:
                return self._project
            raise ValueError("a project is required before starting a project kernel")
        found = self.registry.find(project)
        if found is not None:
            return found
        return self.registry.ensure(project)

    def _config_for(self, snapshot: Path) -> KernelProcessConfig:
        environment = dict(self.config.env or {})
        environment["TST_SNAPSHOT_PATH"] = str(snapshot)
        return replace(self.config, env=environment)

    def _close_project_locked(self) -> None:
        if self._project_client is not None:
            self._project_client.close(graceful=True)
        self._project_client = None
        self._project = None
