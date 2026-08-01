from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tst.kernel.client import StdioKernelClient
from tst.kernel.process import KernelProcessConfig
from tst.protocol.errors import KernelExitedError, KernelOperationError
from tst.routing.router import ActionRouter

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "tst_memory"
_BUILT_KERNEL: Path | None = None


def _kernel_binary() -> Path:
    global _BUILT_KERNEL
    configured = os.environ.get("TST_KERNEL_BIN")
    if configured:
        return Path(configured).expanduser().resolve()
    if _BUILT_KERNEL is not None:
        return _BUILT_KERNEL
    suffix = ".exe" if os.name == "nt" else ""
    subprocess.run(
        ["cargo", "build", "--locked", "--bin", "server"],
        cwd=CRATE,
        check=True,
        capture_output=True,
        text=True,
    )
    _BUILT_KERNEL = CRATE / "target" / "debug" / f"server{suffix}"
    assert _BUILT_KERNEL.is_file()
    return _BUILT_KERNEL


def _config(snapshot: Path) -> KernelProcessConfig:
    return KernelProcessConfig(
        crate_dir=CRATE,
        binary_path=_kernel_binary(),
        startup_timeout=5.0,
        request_timeout=2.0,
        shutdown_timeout=2.0,
        env={
            "TST_SNAPSHOT_PATH": str(snapshot),
            "TST_SNAPSHOT_DEBOUNCE_MS": "5",
        },
    )


def _preference(subject: str, value: str) -> dict[str, object]:
    return {
        "type": "preference",
        "data": {"key": subject, "value": value, "weight": 1.0},
    }


def _payload_value(result: dict[str, object]) -> str | None:
    payload = result.get("payload")
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    preference = (
        data.get("MemoryRecord")
        or data.get("memory_record")
        or data.get("Preference")
        or data.get("preference")
    )
    if isinstance(preference, dict):
        value = preference.get("value")
        return str(value) if value is not None else None
    value = data.get("value")
    return str(value) if value is not None else None


def _memory_record(
    key: str, value: str, source_text: str, *, layer: str = "ltm", created_at: int = 1_000
) -> dict[str, object]:
    return {
        "type": "preference",
        "data": {
            "key": key,
            "value": value,
            "memory_type": "preference",
            "source_text": source_text,
            "created_at": created_at,
            "updated_at": created_at,
            "confidence": 1.0,
            "tags": ["typescript", "programming-language"],
            "source": "user",
            "layer": layer,
            "reinforcement_score": 2.0,
            "deleted": False,
        },
    }


@pytest.mark.integration
def test_python_starts_kernel_and_echoes_request_id(tmp_path: Path) -> None:
    client = StdioKernelClient(_config(tmp_path / "ping.snapshot"))
    client.start()
    try:
        response = client.request("kernel.ping", {})
        assert response.ok is True
        assert response.request_id
        assert response.result == {"pong": True}
        assert response.metrics is not None
        assert response.metrics.kernel_ms is not None
    finally:
        client.close()
    assert client.is_running is False


@pytest.mark.integration
def test_stm_ltm_and_tree_round_trip(tmp_path: Path) -> None:
    with StdioKernelClient(_config(tmp_path / "roundtrip.snapshot")) as client:
        client.store("stm", "session:test:service:name", _preference("service", "Atlas"))
        stm = client.get("stm", "session:test:service:name")
        assert stm["found"] is True
        assert _payload_value(stm) == "Atlas"

        client.store(
            "ltm",
            "user:default:preference:programming_language",
            _preference("programming_language", "TypeScript"),
        )
        ltm = client.get("ltm", "user:default:preference:programming_language")
        assert ltm["found"] is True
        assert _payload_value(ltm) == "TypeScript"

        project = client.tree_insert("Project", "demo")
        source = client.tree_insert("File", "src/service.py", project)
        target = client.tree_insert("Function", "run_route", source)
        caller = client.tree_insert("Function", "main", source)
        client.tree_link(caller, target)
        names = {node["name"] for node in client.tree_query(caller, depth=1)}
        assert {"main", "run_route"} <= names
        found = client.tree_find("run_route")
        assert found and found[0]["node_id"] == target


@pytest.mark.integration
def test_invalid_layer_and_unknown_fields_are_structured_errors(tmp_path: Path) -> None:
    with StdioKernelClient(_config(tmp_path / "errors.snapshot")) as client:
        with pytest.raises(KernelOperationError) as invalid_layer:
            client.request(
                "memory.get",
                {"layer": "archive", "key": "user:default:preference:editor"},
            )
        assert invalid_layer.value.code in {"invalid_params", "invalid_layer"}
        assert invalid_layer.value.request_id

        response = client.request(
            "memory.get",
            {"layer": "ltm", "key": "missing", "unexpected": True},
            raise_on_error=False,
        )
        assert response.ok is False
        assert response.error is not None
        assert response.error.code == "invalid_params"
        deadline = time.monotonic() + 1.0
        while not any(response.request_id in line for line in client.stderr_tail):
            if time.monotonic() >= deadline:
                pytest.fail("kernel error diagnostic omitted the request ID")
            time.sleep(0.01)
        assert all("missing" not in line for line in client.stderr_tail)


@pytest.mark.integration
def test_ltm_survives_restart_but_stm_does_not(tmp_path: Path) -> None:
    snapshot = tmp_path / "restart.snapshot"
    persistent_key = "user:default:preference:editor"
    temporary_key = "session:restart:service:name"

    with StdioKernelClient(_config(snapshot)) as first:
        first.store("ltm", persistent_key, _preference("editor", "Neovim"))
        first.store("stm", temporary_key, _preference("service", "Atlas"))

    assert snapshot.is_file()
    with StdioKernelClient(_config(snapshot)) as second:
        assert _payload_value(second.get("ltm", persistent_key)) == "Neovim"
        assert second.get("stm", temporary_key)["found"] is False


@pytest.mark.integration
def test_metadata_round_trip_and_paraphrased_search(tmp_path: Path) -> None:
    key = "user:default:preference:programming_language"
    with StdioKernelClient(_config(tmp_path / "metadata.snapshot")) as client:
        client.store(
            "ltm",
            key,
            _memory_record(key, "TypeScript", "I prefer TypeScript over JavaScript"),
        )
        exact = client.get("ltm", key)
        payload = exact["payload"]
        assert isinstance(payload, dict)
        data = payload["data"]
        assert isinstance(data, dict)
        record = data["MemoryRecord"]
        assert record["source_text"] == "I prefer TypeScript over JavaScript"
        assert record["tags"] == ["typescript", "programming-language"]

        result = client.search("Which language do I usually use?", layer="ltm", limit=3)
        assert result["matches"]
        assert result["matches"][0]["key"] == key
        assert _payload_value(result["matches"][0]) == "TypeScript"


@pytest.mark.integration
def test_corrupt_latest_snapshot_recovers_previous(tmp_path: Path) -> None:
    snapshot = tmp_path / "recovery.snapshot"
    key = "user:default:preference:editor"
    with StdioKernelClient(_config(snapshot)) as first:
        first.store("ltm", key, _preference("editor", "Vim"))
    with StdioKernelClient(_config(snapshot)) as second:
        second.update("ltm", key, _preference("editor", "Neovim"))

    previous = Path(f"{snapshot}.previous")
    assert previous.is_file()
    snapshot.write_text("{corrupt", encoding="utf-8")

    with StdioKernelClient(_config(snapshot)) as recovered:
        status = recovered.status()
        persistence = status["persistence"]
        assert "previous" in str(persistence.get("recovery", "")).lower()
        assert _payload_value(recovered.get("ltm", key)) == "Vim"


@pytest.mark.integration
def test_repeated_stm_reads_promote_idempotently(tmp_path: Path) -> None:
    key = "session:promotion:preference:language"
    with StdioKernelClient(_config(tmp_path / "promotion.snapshot")) as client:
        client.store("stm", key, _preference("language", "Rust"))
        for _ in range(20):
            client.get("stm", key)
        status = client.status()
        assert status["stm"]["promotions"] == 1
        assert status["ltm"]["entries"] == 1
        promoted = client.get("ltm", key)
        assert promoted["found"] is True
        assert _payload_value(promoted) == "Rust"


@pytest.mark.integration
def test_idle_debounce_saves_without_another_request(tmp_path: Path) -> None:
    snapshot = tmp_path / "idle.snapshot"
    config = _config(snapshot)
    config = KernelProcessConfig(
        crate_dir=config.crate_dir,
        binary_path=config.binary_path,
        startup_timeout=config.startup_timeout,
        request_timeout=config.request_timeout,
        shutdown_timeout=config.shutdown_timeout,
        env={
            "TST_SNAPSHOT_PATH": str(snapshot),
            "TST_SNAPSHOT_DEBOUNCE_MS": "50",
        },
    )
    with StdioKernelClient(config) as client:
        client.store(
            "ltm",
            "user:default:preference:editor",
            _preference("editor", "Neovim"),
        )
        deadline = time.monotonic() + 2.0
        while not snapshot.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert snapshot.exists(), "background debounce did not persist while the kernel was idle"
        assert client.request("persistence.status", {}).result["dirty"] is False


@pytest.mark.integration
def test_structured_update_and_promotion_preserve_metadata(tmp_path: Path) -> None:
    snapshot = tmp_path / "structured.snapshot"
    ltm_key = "user:default:preference:programming_language"
    stm_key = "session:metadata:context:programming_language"
    with StdioKernelClient(_config(snapshot)) as client:
        client.store(
            "ltm",
            ltm_key,
            _memory_record(ltm_key, "Rust", "I prefer Rust", created_at=100),
        )
        client.update(
            "ltm",
            ltm_key,
            _memory_record(ltm_key, "TypeScript", "I now prefer TypeScript", created_at=999),
        )
        updated_payload = client.get("ltm", ltm_key)["payload"]
        updated_record = updated_payload["data"]["MemoryRecord"]
        assert updated_record["created_at"] == 100
        assert updated_record["value"] == "TypeScript"
        assert updated_record["reinforcement_score"] >= 2.0

        client.store(
            "stm",
            stm_key,
            _memory_record(stm_key, "Python", "Use Python for this session", layer="stm"),
        )
        for _ in range(20):
            if not client.get("stm", stm_key)["found"]:
                break
        promoted_payload = client.get("ltm", stm_key)["payload"]
        promoted_record = promoted_payload["data"]["MemoryRecord"]
        assert promoted_record["layer"] == "ltm"
        assert promoted_record["reinforcement_score"] >= 10.0


@pytest.mark.integration
def test_end_to_end_memory_scenarios(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from router.server import run_route

    snapshot = tmp_path / "scenarios.snapshot"
    config = _config(snapshot)
    router = ActionRouter()
    with StdioKernelClient(config) as first:
        stored = run_route(
            "Remember that I prefer TypeScript over JavaScript.",
            client=first,
            action_router=router,
        )
        assert stored.decision.operation == "store"
        temporary = run_route(
            "For this session, call the service Atlas.",
            client=first,
            action_router=router,
        )
        assert temporary.decision.layer == "stm"
        recalled = run_route(
            "What is the service called?",
            client=first,
            action_router=router,
        )
        assert recalled.result["context"] == ["Atlas"]

    with StdioKernelClient(config) as restarted:
        persistent = run_route(
            "Which language should we use for the frontend?",
            client=restarted,
            action_router=ActionRouter(),
        )
        assert persistent.result["context"] == ["TypeScript"]
        temporary = run_route(
            "What is the service called?",
            client=restarted,
            action_router=ActionRouter(),
        )
        assert temporary.result["context"] == []

        run_route(
            "Remember that I use Vim.",
            client=restarted,
            action_router=ActionRouter(),
        )
        forgotten = run_route(
            "Forget my editor preference.",
            client=restarted,
            action_router=ActionRouter(),
        )
        assert forgotten.result["kernel"]["result"]["deleted"] is True
        after_forget = run_route(
            "Which editor do I use?",
            client=restarted,
            action_router=ActionRouter(),
        )
        assert after_forget.result["context"] == []


@pytest.mark.integration
def test_fastapi_route_uses_the_lifecycle_scoped_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from router.server import app

    monkeypatch.setenv("TST_KERNEL_BIN", str(_kernel_binary()))
    monkeypatch.setenv("TST_SNAPSHOT_PATH", str(tmp_path / "fastapi.snapshot"))
    monkeypatch.setenv("TST_ROUTER_SKIP_MODELS", "1")
    with TestClient(app) as web:
        response = web.post(
            "/route",
            json={"query": "Remember that I prefer TypeScript over JavaScript."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"]["operation"] == "store"
        assert body["result"]["kernel"]["result"]["stored"] is True

        status_response = web.get("/status")
        assert status_response.status_code == 200
        service_status = status_response.json()
        assert service_status["kernel"]["protocol_version"] == 1
        assert service_status["router"]["tier_usage"]["deterministic"] == 1
        assert service_status["router"]["operation_layer_confusion"]["store"]["ltm"] == 1


@pytest.mark.integration
@pytest.mark.skipif(os.name == "nt", reason="forced POSIX process termination")
def test_forced_kernel_shutdown_is_reported_as_a_typed_exit(tmp_path: Path) -> None:
    client = StdioKernelClient(_config(tmp_path / "forced.snapshot"))
    client.start()
    assert client.pid is not None
    os.kill(client.pid, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while client.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(KernelExitedError):
        client.request("kernel.ping", {})
    client.close(graceful=False)
