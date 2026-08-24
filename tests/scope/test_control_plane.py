from __future__ import annotations

from pathlib import Path

import pytest

from tst.context import ContextBudget
from tst.kernel.process import KernelProcessConfig
from tst.scope.kernel_manager import KernelManager
from tst.scope.models import Scope
from tst.scope.registry import ProjectRegistry
from tst.service.service import TSTService

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "tst_memory" / "target" / "debug" / "server"


def _manager(registry: ProjectRegistry) -> KernelManager:
    return KernelManager(
        registry,
        config=KernelProcessConfig(
            crate_dir=ROOT / "tst_memory",
            binary_path=KERNEL,
            startup_timeout=5,
            request_timeout=3,
            shutdown_timeout=2,
            env={"TST_SNAPSHOT_DEBOUNCE_MS": "5"},
        ),
    )


@pytest.mark.integration
def test_dual_scope_context_and_restart_lifetimes(tmp_path: Path) -> None:
    pytest.importorskip("pydantic")
    project_root = tmp_path / "demo"
    project_root.mkdir()
    (project_root / "auth.py").write_text(
        "class AuthService:\n    def middleware(self):\n        return True\n",
        encoding="utf-8",
    )
    registry = ProjectRegistry(tmp_path / "home")
    service = TSTService(
        project_root,
        registry=registry,
        kernel_manager=_manager(registry),
        session_id="session-one",
    )
    try:
        global_write = service.store_memory(
            "Prefer typed APIs",
            scope=Scope.GLOBAL,
            subject="api_style",
            memory_type="preference",
        )
        project_write = service.store_memory(
            "Authentication uses JWT middleware",
            scope=Scope.PROJECT,
            subject="auth_architecture",
        )
        service.store_memory(
            "The current task is auth middleware",
            scope=Scope.SESSION,
            subject="current_task",
        )
        pack = service.retrieve_context("authentication middleware", budget=ContextBudget(max_tokens=500))
        assert pack.project == "demo"
        assert any(item.scope is Scope.PROJECT for item in pack.items)
        assert any(item.source == "tree" for item in pack.items)
        assert global_write["scope"] == "global"
        assert project_write["scope"] == "project"
        assert service.kernel_manager.global_process is not None
        assert service.kernel_manager.active_project_process is not None
        assert service.kernel_manager.global_process.pid != service.kernel_manager.active_project_process.pid

        moved = service.move_memory(project_write["key"], Scope.GLOBAL)
        assert moved["moved"] is True
        assert not any(item["key"] == project_write["key"] for item in service.list_memories(Scope.PROJECT))
        assert any(item["key"] == moved["key"] for item in service.list_memories(Scope.GLOBAL))
    finally:
        service.close()

    restarted = TSTService(
        project_root,
        registry=registry,
        kernel_manager=_manager(registry),
        session_id="session-two",
    )
    try:
        assert service.project.id == restarted.project.id
        assert restarted.list_memories(Scope.GLOBAL)
        assert restarted.list_memories(Scope.PROJECT) == []
        assert restarted.list_memories(Scope.SESSION) == []
    finally:
        restarted.close()


@pytest.mark.integration
def test_project_switch_keeps_memory_and_tree_local(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "first.py").write_text("class FirstService:\n    pass\n", encoding="utf-8")
    (second_root / "second.py").write_text("class SecondService:\n    pass\n", encoding="utf-8")
    registry = ProjectRegistry(tmp_path / "home")
    service = TSTService(first_root, registry=registry, kernel_manager=_manager(registry), session_id="switch")
    try:
        service.store_memory("first project only", scope=Scope.PROJECT, subject="local_fact")
        service.index_project()
        service.switch_project(second_root)
        assert service.list_memories(Scope.PROJECT) == []
        service.store_memory("second project only", scope=Scope.PROJECT, subject="local_fact")
        service.index_project()
        assert service.find_symbol("FirstService") == []

        service.switch_project(first_root)
        assert [item["value"] for item in service.list_memories(Scope.PROJECT)] == ["first project only"]
        service.index_project()
        assert service.find_symbol("FirstService")
        assert service.find_symbol("SecondService") == []
    finally:
        service.close()
