#!/usr/bin/env python3
"""Capture a machine-readable TST baseline through the shared kernel client."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import TypeVar

from tst.analysis import IncrementalIndexer
from tst.evaluation.metrics import percentile
from tst.kernel.client import StdioKernelClient
from tst.kernel.process import KernelProcessConfig, build_kernel, default_kernel_binary
from tst.routing.router import ActionRouter

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "tst_memory"
T = TypeVar("T")


def _timed(call: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - started) * 1_000.0


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean_ms": mean(values) if values else 0.0,
        "p50_ms": median(values) if values else 0.0,
        "p95_ms": percentile(values, 95),
    }


def _payload(value: str) -> dict[str, object]:
    return {
        "type": "preference",
        "data": {"key": "baseline", "value": value, "weight": 1.0},
    }


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _version(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else "unavailable"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(128 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip()) * 1024
    except ValueError:
        return None


def _rust_tests() -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        ["cargo", "test", "--locked", "--all-targets"],
        cwd=CRATE,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
        "stdout_tail": result.stdout[-2_000:],
        "stderr_tail": result.stderr[-2_000:],
    }


def _rust_stress() -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        ["cargo", "run", "--locked", "--release", "--bin", "stress"],
        cwd=CRATE,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
        "stdout_tail": result.stdout[-8_000:],
        "stderr_tail": result.stderr[-2_000:],
    }


def capture(
    binary: Path,
    *,
    include_rust_tests: bool,
    include_rust_stress: bool,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": 2,
        "captured_at_unix_ms": int(time.time() * 1_000),
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "rustc": _version(["rustc", "--version"]),
            "cargo": _version(["cargo", "--version"]),
        },
        "kernel_binary": {
            "path": str(binary),
            "size_bytes": binary.stat().st_size,
            "sha256": _file_sha256(binary),
        },
        "commands": {
            "rust_tests": "cargo test --locked --all-targets",
            "rust_stress": "cargo run --locked --release --bin stress",
            "routing": "python scripts/evaluate_routing.py",
            "retrieval": "python scripts/evaluate_retrieval.py",
            "baseline": "python scripts/baseline.py",
        },
    }
    if include_rust_tests:
        document["rust_tests"] = _rust_tests()
    if include_rust_stress:
        document["rust_stress"] = _rust_stress()

    with tempfile.TemporaryDirectory(prefix="tst-baseline-") as directory:
        snapshot = Path(directory) / "ltm.snapshot"
        config = KernelProcessConfig(
            crate_dir=CRATE,
            binary_path=binary,
            startup_timeout=10.0,
            request_timeout=5.0,
            env={
                "TST_SNAPSHOT_PATH": str(snapshot),
                "TST_SNAPSHOT_DEBOUNCE_MS": "10",
            },
        )
        client = StdioKernelClient(config)
        _, startup_ms = _timed(client.start)
        protocol_total: list[float] = []
        protocol_kernel: list[float] = []
        stm_write: list[float] = []
        stm_read: list[float] = []
        ltm_write: list[float] = []
        ltm_read: list[float] = []
        search: list[float] = []
        tree_query: list[float] = []
        tree_find: list[float] = []
        snapshot_save: list[float] = []
        rss_samples: list[int] = []
        try:
            initial_rss = _rss_bytes(client.pid)
            if initial_rss is not None:
                rss_samples.append(initial_rss)
            for _ in range(100):
                response, elapsed = _timed(lambda: client.request("kernel.ping", {}))
                protocol_total.append(elapsed)
                protocol_kernel.append(
                    response.metrics.kernel_ms
                    if response.metrics and response.metrics.kernel_ms is not None
                    else 0.0
                )

            for index in range(100):
                key = f"session:baseline:turn:{index}"
                _, elapsed = _timed(
                    lambda key=key, index=index: client.store(
                        "stm", key, _payload(f"temporary-{index}")
                    )
                )
                stm_write.append(elapsed)
            for index in range(100):
                _, elapsed = _timed(
                    lambda index=index: client.get("stm", f"session:baseline:turn:{index}")
                )
                stm_read.append(elapsed)

            for index in range(100):
                key = f"user:baseline:preference:item:{index}"
                _, elapsed = _timed(
                    lambda key=key, index=index: client.store(
                        "ltm", key, _payload(f"persistent-{index}")
                    )
                )
                ltm_write.append(elapsed)
            for index in range(100):
                _, elapsed = _timed(
                    lambda index=index: client.get("ltm", f"user:baseline:preference:item:{index}")
                )
                ltm_read.append(elapsed)
            for _ in range(20):
                _, elapsed = _timed(
                    lambda: client.search(
                        "persistent preference",
                        layer="ltm",
                        prefix="user:baseline:preference",
                        limit=5,
                    )
                )
                search.append(elapsed)

            for _ in range(5):
                _, elapsed = _timed(lambda: client.request("persistence.save", {}))
                snapshot_save.append(elapsed)

            client.tree_clear()
            project = client.tree_insert("Project", "baseline")
            file_node = client.tree_insert("File", "src/main.py", project)
            for index in range(50):
                client.tree_insert("Function", f"function_{index}", file_node)
            for _ in range(20):
                _, elapsed = _timed(lambda: client.tree_query(file_node, depth=1))
                tree_query.append(elapsed)
                _, elapsed = _timed(lambda: client.tree_find("function_25", limit=10))
                tree_find.append(elapsed)
            final_rss = _rss_bytes(client.pid)
            if final_rss is not None:
                rss_samples.append(final_rss)
            status_before_restart = client.status()
        finally:
            client.close(graceful=True)

        restarted = StdioKernelClient(config)
        _, restart_ms = _timed(restarted.start)
        try:
            persistent_after_restart = restarted.get("ltm", "user:baseline:preference:item:0")
            temporary_after_restart = restarted.get("stm", "session:baseline:turn:0")
            persistence = restarted.request("persistence.status", {}).result or {}
        finally:
            restarted.close(graceful=True)

        document["kernel"] = {
            "startup_ms": startup_ms,
            "restart_ms": restart_ms,
            "protocol_round_trip": _latency(protocol_total),
            "protocol_kernel": _latency(protocol_kernel),
            "protocol_overhead": _latency(
                [
                    max(0.0, total - kernel)
                    for total, kernel in zip(protocol_total, protocol_kernel, strict=False)
                ]
            ),
            "stm_write": _latency(stm_write),
            "stm_read": _latency(stm_read),
            "ltm_write": _latency(ltm_write),
            "ltm_read": _latency(ltm_read),
            "memory_search": _latency(search),
            "snapshot_save": _latency(snapshot_save),
            "tree_find": _latency(tree_find),
            "tree_query": _latency(tree_query),
            "process_rss_bytes": {
                "samples": rss_samples,
                "maximum": max(rss_samples, default=0),
            },
            "status": status_before_restart,
            "restart_persistence": {
                "ltm_found": bool(persistent_after_restart.get("found")),
                "stm_found": bool(temporary_after_restart.get("found")),
                "persistence": persistence,
            },
        }

    analysis_started = time.perf_counter()
    indexer = IncrementalIndexer(ROOT / "test_project")
    first_index = indexer.index()
    second_index = indexer.index()
    document["analysis"] = {
        "elapsed_ms": (time.perf_counter() - analysis_started) * 1_000.0,
        "nodes": len(indexer.graph.nodes),
        "edges": len(indexer.graph.edges),
        "first_parsed_files": first_index.parsed_files,
        "second_parsed_files": second_index.parsed_files,
        "unchanged_files": len(second_index.unchanged),
        "unchanged_ms_per_file": (second_index.elapsed_ms / max(1, len(second_index.unchanged))),
        "parse_errors": first_index.parse_errors,
    }

    router = ActionRouter()
    routing_samples = (
        "Remember that I prefer TypeScript.",
        "For this conversation, call the service Atlas.",
        "What did I just say?",
        "What did I say about my editor?",
        "/analyze test_project",
        "Explain a binary search.",
    )
    routing_latencies: list[float] = []
    routing_results: list[dict[str, object]] = []
    for query in routing_samples:
        decision, elapsed = _timed(lambda query=query: router.route(query))
        routing_latencies.append(elapsed)
        routing_results.append({"query": query, "decision": decision.model_dump(mode="json")})
    document["routing"] = {
        "latency": _latency(routing_latencies),
        "parse_failures": router.parse_failures,
        "samples": routing_results,
    }
    kernel = document["kernel"]
    analysis = document["analysis"]
    assert isinstance(kernel, dict) and isinstance(analysis, dict)
    gates = {
        "stm_read_p95_under_1_ms": kernel["stm_read"]["p95_ms"] < 1.0,
        "ltm_read_p95_under_5_ms": kernel["ltm_read"]["p95_ms"] < 5.0,
        "memory_search_p95_under_20_ms": kernel["memory_search"]["p95_ms"] < 20.0,
        "tree_find_p95_under_20_ms": kernel["tree_find"]["p95_ms"] < 20.0,
        "tree_query_p95_under_50_ms": kernel["tree_query"]["p95_ms"] < 50.0,
        "snapshot_save_p95_under_250_ms": kernel["snapshot_save"]["p95_ms"] < 250.0,
        "protocol_overhead_p95_under_2_ms": kernel["protocol_overhead"]["p95_ms"] < 2.0,
        "incremental_unchanged_under_2_ms_per_file": analysis["unchanged_ms_per_file"] < 2.0,
        "ltm_restart_persisted": kernel["restart_persistence"]["ltm_found"],
        "stm_restart_cleared": not kernel["restart_persistence"]["stm_found"],
    }
    document["gates"] = {"passed": all(gates.values()), "checks": gates}
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("baseline-results.json"))
    parser.add_argument("--kernel-bin", type=Path)
    parser.add_argument("--skip-rust-tests", action="store_true")
    parser.add_argument("--skip-rust-stress", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    binary = (args.kernel_bin or default_kernel_binary(CRATE)).expanduser().resolve()
    if not binary.is_file():
        print("Release kernel missing; building it once for the baseline...", file=sys.stderr)
        binary = build_kernel(KernelProcessConfig(crate_dir=CRATE))
    document = capture(
        binary,
        include_rust_tests=not args.skip_rust_tests,
        include_rust_stress=not args.skip_rust_stress,
    )
    args.output.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    print(f"Saved baseline to {args.output}")
    checks_ok = bool(document["gates"]["passed"])
    rust_ok = all(
        bool(document[name]["ok"]) for name in ("rust_tests", "rust_stress") if name in document
    )
    return 0 if args.report_only or (checks_ok and rust_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
