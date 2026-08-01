#!/usr/bin/env python3
"""Measure repository indexing and symbol-centered Tree retrieval."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any

from tst.analysis import IncrementalIndexer
from tst.evaluation.metrics import percentile
from tst.kernel.client import StdioKernelClient
from tst.kernel.process import KernelProcessConfig


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "p50_ms": median(values) if values else 0.0,
        "p95_ms": percentile(values, 95),
    }


def _sync_tree(client: StdioKernelClient, indexer: IncrementalIndexer) -> dict[int, int]:
    client.tree_clear()
    graph = indexer.graph
    mapped: dict[int, int] = {}
    pending = dict(graph.nodes)
    node_types = {
        "method": "function",
        "variable": "symbol",
        "external_module": "external",
        "external_symbol": "external",
    }
    while pending:
        progressed = False
        for node_id, node in list(pending.items()):
            if node.parent is not None and node.parent not in mapped:
                continue
            mapped[node_id] = client.tree_insert(
                node_types.get(node.node_type, node.node_type),
                node.name,
                mapped.get(node.parent) if node.parent is not None else None,
                qualified_name=node.qualified_name,
                file_path=node.file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                signature=node.signature,
                content_hash=node.content_hash,
            )
            del pending[node_id]
            progressed = True
        if not progressed:
            raise RuntimeError("code graph contains an unresolved parent cycle")
    for edge in graph.edges:
        client.tree_link(
            mapped[edge.source],
            mapped[edge.target],
            edge_type=edge.kind.value,
            confidence=edge.confidence,
        )
    return mapped


def run_benchmark(
    root: Path,
    *,
    kernel_config: KernelProcessConfig | None = None,
) -> dict[str, Any]:
    indexer = IncrementalIndexer(root)
    started = time.perf_counter()
    first = indexer.index()
    first_ms = (time.perf_counter() - started) * 1_000.0
    started = time.perf_counter()
    unchanged = indexer.index()
    unchanged_ms = (time.perf_counter() - started) * 1_000.0

    queryable = [
        node
        for node in indexer.graph.nodes.values()
        if node.node_type not in {"project", "external_module", "external_symbol"}
    ][:200]
    query_latencies: list[float] = []
    for node in queryable:
        started = time.perf_counter()
        indexer.graph.query(node.node_id, depth=1, max_nodes=40, token_budget=2_000)
        query_latencies.append((time.perf_counter() - started) * 1_000.0)

    per_file_unchanged = unchanged_ms / max(1, len(unchanged.unchanged))
    document: dict[str, Any] = {
        "schema_version": 1,
        "root": str(root),
        "index": {
            "first_ms": first_ms,
            "parsed_files": first.parsed_files,
            "parse_errors": first.parse_errors,
            "nodes": len(indexer.graph.nodes),
            "edges": len(indexer.graph.edges),
            "unchanged_ms": unchanged_ms,
            "unchanged_files": len(unchanged.unchanged),
            "unchanged_ms_per_file": per_file_unchanged,
        },
        "python_graph_query": _latency(query_latencies),
    }

    if kernel_config is not None:
        client = StdioKernelClient(kernel_config)
        client.start()
        try:
            mapped = _sync_tree(client, indexer)
            find_latencies: list[float] = []
            subgraph_latencies: list[float] = []
            for node in queryable:
                started = time.perf_counter()
                client.tree_find(node.qualified_name, limit=10)
                find_latencies.append((time.perf_counter() - started) * 1_000.0)
                started = time.perf_counter()
                client.tree_query(mapped[node.node_id], depth=1, max_nodes=40, token_budget=2_000)
                subgraph_latencies.append((time.perf_counter() - started) * 1_000.0)
            document["kernel_tree_find"] = _latency(find_latencies)
            document["kernel_tree_query"] = _latency(subgraph_latencies)
            document["kernel_status"] = client.status()["tree"]
        finally:
            client.close(graceful=True)

    gates = {
        "incremental_unchanged_per_file": per_file_unchanged < 2.0,
        "python_graph_query_p95": document["python_graph_query"]["p95_ms"] < 50.0,
    }
    if "kernel_tree_find" in document:
        gates["kernel_tree_find_p95"] = document["kernel_tree_find"]["p95_ms"] < 20.0
        gates["kernel_tree_query_p95"] = document["kernel_tree_query"]["p95_ms"] < 50.0
    document["gates"] = {"passed": all(gates.values()), "checks": gates}
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path("test_project"))
    parser.add_argument("--output", type=Path, default=Path("layer4-results.json"))
    parser.add_argument("--with-kernel", action="store_true")
    parser.add_argument("--build-kernel", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    root = args.path.expanduser().resolve(strict=True)
    kernel_config = None
    with tempfile.TemporaryDirectory(prefix="tst-layer4-") as directory:
        if args.with_kernel:
            kernel_config = KernelProcessConfig(
                build_kernel=args.build_kernel,
                env={"TST_SNAPSHOT_PATH": str(Path(directory) / "ltm.snapshot")},
            )
        document = run_benchmark(root, kernel_config=kernel_config)
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))
    return 0 if args.report_only or document["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
