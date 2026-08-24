from __future__ import annotations

from pathlib import Path

import pytest

from tst.kernel.process import KernelProcessConfig
from tst.scope.kernel_manager import KernelManager
from tst.scope.registry import ProjectRegistry
from tst.service.service import TSTService

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "tst_memory" / "target" / "debug" / "server"


@pytest.mark.integration
def test_control_plane_http_surface(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ProjectRegistry(tmp_path / "home")
    project = tmp_path / "project"
    project.mkdir()
    (project / "service.py").write_text("class AuthService:\n    pass\n", encoding="utf-8")
    manager = KernelManager(
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
    service = TSTService(project, registry=registry, kernel_manager=manager, session_id="api-session")
    try:
        from tst.server.app import create_app

        with TestClient(create_app(service=service)) as web:
            page = web.get("/")
            assert page.status_code == 200
            assert "TST Control Plane" in page.text
            assert web.get("/assets/index-BhwV92ui.js").status_code == 200
            stored = web.post(
                "/api/v1/memories",
                json={"value": "Authentication uses JWT", "scope": "project", "subject": "auth"},
            )
            assert stored.status_code == 200
            assert stored.json()["stored"] is True

            context = web.post(
                "/api/v1/context/preview",
                json={"query": "authentication", "budget": 200},
            )
            assert context.status_code == 200
            assert context.json()["items"]

            events = web.get("/api/v1/events/stream")
            assert events.status_code == 200
            assert "memory.store" in events.text
            assert "Authentication uses JWT" not in events.text
    finally:
        service.close()
