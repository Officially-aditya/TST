"""Protocol-overhead benchmark using the shared v1 STDIO client."""

from __future__ import annotations

import time

from tst.kernel.client import StdioKernelClient
from tst.kernel.process import KernelProcessConfig


def _payload(index: int) -> dict:
    return {
        "type": "token_stats",
        "data": {
            "key": f"benchmark:ltm:item:{index}",
            "value": f"data_{index}",
            "memory_type": "benchmark",
            "source_text": f"data_{index}",
            "created_at": 0,
            "updated_at": 0,
            "confidence": 1.0,
            "tags": ["benchmark"],
            "source": "evaluation",
            "layer": "ltm",
            "reinforcement_score": 0.0,
            "deleted": False,
        },
    }


def test_rust_kernel_latency(num_items: int = 10_000) -> None:
    print("Building TST Memory Kernel for Profiling (explicit benchmark build)...")
    client = StdioKernelClient(KernelProcessConfig(build_kernel=True, request_timeout=10))
    client.start()
    try:
        print("\n--- Latency Benchmark ---")
        started = time.perf_counter()
        for index in range(num_items):
            key = f"benchmark:ltm:item:{index}"
            client.store("ltm", key, _payload(index))
        write_seconds = time.perf_counter() - started
        print(f"Write Throughput: {num_items / write_seconds:.2f} requests/sec")
        print(f"Avg Write Latency: {(write_seconds / num_items) * 1000:.3f} ms")

        started = time.perf_counter()
        for index in range(num_items):
            client.get("ltm", f"benchmark:ltm:item:{index}")
        read_seconds = time.perf_counter() - started
        print(f"Read Throughput: {num_items / read_seconds:.2f} requests/sec")
        print(f"Avg Read Latency: {(read_seconds / num_items) * 1000:.3f} ms")
    finally:
        client.close(graceful=True)


if __name__ == "__main__":
    test_rust_kernel_latency()
