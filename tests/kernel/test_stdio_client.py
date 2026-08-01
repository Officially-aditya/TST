from __future__ import annotations

import sys

import pytest

from tst.kernel.client import StdioKernelClient
from tst.kernel.process import KernelProcessConfig
from tst.protocol.errors import (
    KernelExitedError,
    KernelNotStartedError,
    KernelOperationError,
    KernelProtocolError,
    KernelTimeoutError,
)

FAKE_KERNEL = r"""
import json, sys
print("READY", flush=True)
for line in sys.stdin:
    req = json.loads(line)
    op = req["operation"]
    if op == "kernel.shutdown":
        result = {"shutdown": True}
    elif op == "kernel.ping":
        result = {"status": "ok"}
    elif op == "memory.get":
        result = {"found": True, **req["params"], "payload": {"value": "TypeScript"}}
    elif op == "memory.delete":
        response = {
            "protocol_version": 1,
            "request_id": req["request_id"],
            "ok": False,
            "result": None,
            "error": {"code": "protected", "message": "reserved key", "details": None, "retryable": False},
            "metrics": {"kernel_ms": 0.01},
        }
        print(json.dumps(response), flush=True)
        continue
    else:
        result = {"echo_operation": op, **req["params"]}
    response = {
        "protocol_version": 1,
        "request_id": req["request_id"],
        "ok": True,
        "result": result,
        "error": None,
        "metrics": {"kernel_ms": 0.01},
    }
    print(json.dumps(response), flush=True)
    if op == "kernel.shutdown":
        break
"""


@pytest.fixture
def client(tmp_path):
    config = KernelProcessConfig(
        crate_dir=tmp_path,
        command=(sys.executable, "-u", "-c", FAKE_KERNEL),
        startup_timeout=2,
        request_timeout=2,
    )
    kernel = StdioKernelClient(config)
    kernel.start()
    try:
        yield kernel
    finally:
        kernel.close()


def test_request_round_trip_echoes_id_and_metrics(client):
    response = client.request("kernel.ping", {})
    assert response.ok is True
    assert response.result == {"status": "ok"}
    assert response.request_id
    assert response.metrics.kernel_ms == pytest.approx(0.01)


def test_convenience_memory_get_uses_versioned_request(client):
    result = client.get("ltm", "user:default:preference:programming_language")
    assert result["found"] is True
    assert result["layer"] == "ltm"
    assert result["payload"]["value"] == "TypeScript"


def test_structured_kernel_error_becomes_typed_exception(client):
    with pytest.raises(KernelOperationError) as caught:
        client.delete("ltm", "system:secret")
    assert caught.value.code == "protected"
    assert caught.value.request_id


def test_close_requests_graceful_shutdown(client):
    pid = client.pid
    client.close(graceful=True)
    assert pid is not None
    assert client.is_running is False


def _script_client(tmp_path, script: str, **config_values) -> StdioKernelClient:
    config = KernelProcessConfig(
        crate_dir=tmp_path,
        command=(sys.executable, "-u", "-c", script),
        startup_timeout=config_values.pop("startup_timeout", 1.0),
        request_timeout=config_values.pop("request_timeout", 1.0),
        shutdown_timeout=config_values.pop("shutdown_timeout", 0.1),
        **config_values,
    )
    return StdioKernelClient(config)


def test_request_before_start_is_typed(tmp_path):
    kernel = _script_client(tmp_path, FAKE_KERNEL)
    with pytest.raises(KernelNotStartedError):
        kernel.request("kernel.ping", {})


def test_startup_and_request_timeouts_terminate_ambiguous_process(tmp_path):
    startup = _script_client(
        tmp_path,
        "import time; time.sleep(10)",
        startup_timeout=0.05,
    )
    with pytest.raises(KernelTimeoutError):
        startup.start()
    assert startup.is_running is False

    request = _script_client(
        tmp_path,
        'import sys,time; print("READY", flush=True); sys.stdin.readline(); time.sleep(10)',
        request_timeout=0.05,
    )
    request.start()
    with pytest.raises(KernelTimeoutError):
        request.request("kernel.ping", {})
    assert request.is_running is False


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        '{"protocol_version":1,"request_id":"wrong","ok":true,"result":{},'
        '"error":null,"metrics":{"kernel_ms":0}}',
    ],
)
def test_malformed_and_mismatched_responses_are_protocol_errors(tmp_path, response):
    script = (
        'import sys; print("READY", flush=True); sys.stdin.readline(); '
        f"print({response!r}, flush=True); sys.stdin.readline()"
    )
    kernel = _script_client(tmp_path, script)
    kernel.start()
    with pytest.raises(KernelProtocolError):
        kernel.request("kernel.ping", {})
    assert kernel.is_running is False


def test_process_exit_preserves_stderr_diagnostics(tmp_path):
    script = (
        'import sys; print("READY", flush=True); sys.stdin.readline(); '
        'print("fatal kernel detail", file=sys.stderr, flush=True); raise SystemExit(7)'
    )
    kernel = _script_client(tmp_path, script)
    kernel.start()
    with pytest.raises(KernelExitedError):
        kernel.request("kernel.ping", {})
    assert "fatal kernel detail" in " ".join(kernel.stderr_tail)


def test_oversized_response_is_rejected_and_reader_state_can_restart(tmp_path):
    oversized_script = (
        'import sys; print("READY", flush=True); sys.stdin.readline(); '
        'print("x" * 500, flush=True); sys.stdin.readline()'
    )
    kernel = _script_client(tmp_path, oversized_script, max_response_bytes=128)
    kernel.start()
    with pytest.raises(KernelProtocolError, match="exceeded"):
        kernel.request("kernel.ping", {})
    assert kernel.is_running is False

    healthy = _script_client(tmp_path, FAKE_KERNEL)
    healthy.start()
    healthy.close()
    healthy.start()
    assert healthy.ping() == {"status": "ok"}
    healthy.close()


def test_process_configuration_rejects_non_positive_limits(tmp_path):
    with pytest.raises(ValueError):
        KernelProcessConfig(crate_dir=tmp_path, request_timeout=0)
    with pytest.raises(ValueError):
        KernelProcessConfig(crate_dir=tmp_path, max_response_bytes=0)
