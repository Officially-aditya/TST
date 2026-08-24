"""Thread-safe synchronous client for the versioned kernel NDJSON protocol."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from tst.protocol.errors import (
    KernelExitedError,
    KernelNotStartedError,
    KernelOperationError,
    KernelProtocolError,
    KernelStartError,
    KernelTimeoutError,
)
from tst.protocol.models import KernelRequest, KernelResponse
from tst.protocol.operations import MemoryLayer, ProtocolOperation

from .process import KernelProcessConfig, build_kernel

_STREAM_CLOSED = object()


class _OversizedResponse:
    def __init__(self, size: int) -> None:
        self.size = size


class StdioKernelClient:
    """Own a Rust kernel process and exchange one NDJSON envelope at a time.

    Public API:

    * ``start() -> None``
    * ``request(operation, params, timeout=None) -> KernelResponse``
    * ``close(graceful=True) -> None``

    Every request receives a generated ID, validates the echoed response ID,
    and is serialized behind a lock. A timeout makes the stream ambiguous, so
    the client terminates that process instead of risking response mix-ups.
    """

    def __init__(self, config: KernelProcessConfig | None = None) -> None:
        self.config = config or KernelProcessConfig()
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | object] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=self.config.stderr_history)
        self._request_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.is_running:
                return
            if self.config.build_kernel:
                build_kernel(self.config)

            command = self.config.resolved_command()
            if self.config.command is None:
                binary = Path(command[0])
                if not binary.is_file():
                    raise KernelStartError(
                        f"kernel binary not found: {binary}. "
                        "Build it explicitly with `cargo build --release --bin server` "
                        "or start with the developer build option."
                    )

            child_env = os.environ.copy()
            if self.config.env:
                child_env.update(self.config.env)

            self._stdout_queue = queue.Queue()
            output_queue = self._stdout_queue
            self._stderr.clear()
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=self.config.resolved_working_directory(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    env=child_env,
                )
            except OSError as exc:
                self._process = None
                raise KernelStartError(f"could not start kernel: {exc}") from exc

            assert self._process.stdout is not None
            assert self._process.stderr is not None
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                args=(self._process.stdout, output_queue),
                name="tst-kernel-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(self._process.stderr,),
                name="tst-kernel-stderr",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            try:
                ready = self._next_stdout(self.config.startup_timeout, phase="startup")
            except Exception:
                self._terminate_process()
                raise
            if ready.strip() != "READY":
                self._terminate_process()
                raise KernelStartError(
                    f"kernel emitted {ready!r} before READY; stderr: {self._stderr_summary()}"
                )

    def request(
        self,
        operation: ProtocolOperation | str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        raise_on_error: bool = True,
    ) -> KernelResponse:
        request = KernelRequest.create(operation, params)
        deadline = timeout if timeout is not None else self.config.request_timeout

        with self._request_lock:
            process = self._require_running()
            assert process.stdin is not None
            wire = request.model_dump_json() + "\n"
            try:
                process.stdin.write(wire)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._terminate_process()
                raise KernelExitedError(
                    f"kernel closed stdin while sending {request.request_id}: {exc}"
                ) from exc

            try:
                raw = self._next_stdout(deadline, phase=f"request {request.request_id}")
            except KernelTimeoutError:
                self._terminate_process()
                raise

            try:
                response = KernelResponse.model_validate_json(raw)
            except (ValidationError, ValueError) as exc:
                self._terminate_process()
                raise KernelProtocolError(
                    f"invalid response for {request.request_id}: {raw[:500]!r}"
                ) from exc

            if response.request_id != request.request_id:
                self._terminate_process()
                raise KernelProtocolError(
                    f"response ID {response.request_id!r} did not match "
                    f"request ID {request.request_id!r}"
                )
            if not response.ok and raise_on_error:
                assert response.error is not None
                raise KernelOperationError(
                    response.error.code,
                    response.error.message,
                    request_id=response.request_id,
                    details=response.error.details,
                    retryable=response.error.retryable,
                )
            return response

    def close(self, graceful: bool = True) -> None:
        with self._lifecycle_lock:
            process = self._process
            if process is None:
                return

            if graceful and process.poll() is None:
                try:
                    self.request(
                        ProtocolOperation.KERNEL_SHUTDOWN,
                        {},
                        timeout=self.config.shutdown_timeout,
                        raise_on_error=False,
                    )
                except Exception:
                    # Shutdown remains best-effort; termination below is bounded.
                    pass

            if process.poll() is None:
                try:
                    process.wait(timeout=self.config.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
            self._close_streams(process)
            self._join_reader_threads()
            self._process = None

    def ping(self) -> dict[str, Any]:
        return self._result(ProtocolOperation.KERNEL_PING, {})

    def status(self) -> dict[str, Any]:
        return self._result(ProtocolOperation.KERNEL_STATUS, {})

    def store(
        self,
        layer: MemoryLayer | str,
        key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._result(
            ProtocolOperation.MEMORY_STORE,
            {"layer": str(layer), "key": key, "payload": payload},
        )

    def get(self, layer: MemoryLayer | str, key: str) -> dict[str, Any]:
        return self._result(
            ProtocolOperation.MEMORY_GET,
            {"layer": str(layer), "key": key},
        )

    def update(
        self,
        layer: MemoryLayer | str,
        key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._result(
            ProtocolOperation.MEMORY_UPDATE,
            {"layer": str(layer), "key": key, "payload": payload},
        )

    def delete(self, layer: MemoryLayer | str, key: str) -> dict[str, Any]:
        return self._result(
            ProtocolOperation.MEMORY_DELETE,
            {"layer": str(layer), "key": key},
        )

    def search(
        self,
        query: str,
        *,
        layer: MemoryLayer | str | None = None,
        prefix: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query, "limit": limit}
        if layer is not None:
            params["layer"] = str(layer)
        if prefix is not None:
            params["prefix"] = prefix
        return self._result(ProtocolOperation.MEMORY_SEARCH, params)

    def tree_clear(self) -> dict[str, Any]:
        return self._result(ProtocolOperation.TREE_CLEAR, {})

    def tree_insert(
        self,
        node_type: str,
        name: str,
        parent_id: int | None = None,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        signature: str | None = None,
        content_hash: str | None = None,
    ) -> int:
        params: dict[str, Any] = {"node_type": node_type, "name": name}
        if parent_id is not None:
            params["parent_id"] = parent_id
        for key, value in {
            "qualified_name": qualified_name,
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "signature": signature,
            "content_hash": content_hash,
        }.items():
            if value is not None:
                params[key] = value
        return int(self._result(ProtocolOperation.TREE_INSERT, params)["node_id"])

    def tree_remove(self, node_id: int) -> dict[str, Any]:
        return self._result(ProtocolOperation.TREE_REMOVE, {"node_id": node_id})

    def tree_link(
        self,
        source_id: int,
        target_id: int,
        *,
        edge_type: str = "references",
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        return self._result(
            ProtocolOperation.TREE_LINK,
            {
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "confidence": confidence,
            },
        )

    def tree_unlink(self, source_id: int, target_id: int) -> dict[str, Any]:
        return self._result(
            ProtocolOperation.TREE_UNLINK,
            {"source_id": source_id, "target_id": target_id},
        )

    def tree_query(
        self,
        node_id: int,
        depth: int = 3,
        *,
        max_nodes: int = 100,
        token_budget: int = 2_000,
    ) -> list[dict[str, Any]]:
        result = self._result(
            ProtocolOperation.TREE_QUERY,
            {
                "node_id": node_id,
                "depth": depth,
                "max_nodes": max_nodes,
                "token_budget": token_budget,
            },
        )
        return list(result.get("nodes", []))

    def tree_find(self, name: str, limit: int = 10) -> list[dict[str, Any]]:
        result = self._result(
            ProtocolOperation.TREE_FIND,
            {"name": name, "limit": limit},
        )
        return list(result.get("nodes", []))

    def __enter__(self) -> StdioKernelClient:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close(graceful=True)

    # Compatibility name for older callers while they migrate to close().
    stop = close

    def _result(self, operation: ProtocolOperation, params: dict[str, Any]) -> dict[str, Any]:
        response = self.request(operation, params)
        return response.result or {}

    def _require_running(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None:
            raise KernelNotStartedError("kernel client has not been started")
        return_code = process.poll()
        if return_code is not None:
            self._process = None
            raise KernelExitedError(
                f"kernel exited with status {return_code}; stderr: {self._stderr_summary()}"
            )
        return process

    def _next_stdout(self, timeout: float, *, phase: str) -> str:
        try:
            item = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise KernelTimeoutError(f"kernel {phase} timed out after {timeout:.3f}s") from exc
        if item is _STREAM_CLOSED:
            process = self._process
            code = process.poll() if process is not None else None
            raise KernelExitedError(
                f"kernel stdout closed during {phase} (status {code}); "
                f"stderr: {self._stderr_summary()}"
            )
        if isinstance(item, _OversizedResponse):
            self._terminate_process()
            raise KernelProtocolError(
                f"kernel {phase} response exceeded the configured "
                f"{self.config.max_response_bytes} byte limit ({item.size} bytes)"
            )
        assert isinstance(item, str)
        return item

    def _read_stdout(self, stream: TextIO, output_queue: queue.Queue[str | object]) -> None:
        try:
            for line in stream:
                encoded_size = len(line.encode("utf-8"))
                if encoded_size > self.config.max_response_bytes:
                    output_queue.put(_OversizedResponse(encoded_size))
                    return
                output_queue.put(line.rstrip("\r\n"))
        finally:
            output_queue.put(_STREAM_CLOSED)

    def _read_stderr(self, stream: TextIO) -> None:
        for line in stream:
            self._stderr.append(line.rstrip("\r\n"))

    def _stderr_summary(self) -> str:
        return " | ".join(tuple(self._stderr)[-10:]) or "(empty)"

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        self._close_streams(process)
        self._join_reader_threads()
        self._process = None

    def _join_reader_threads(self) -> None:
        current = threading.current_thread()
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=1.0)
        self._stdout_thread = None
        self._stderr_thread = None

    @staticmethod
    def _close_streams(process: subprocess.Popen[str]) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
