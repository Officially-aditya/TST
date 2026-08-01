"""STM/LTM latency and restart-persistence benchmarks over protocol v1."""

from __future__ import annotations

import time

from tst.kernel.client import StdioKernelClient


def _record(key: str, value: str, layer: str, memory_type: str) -> dict:
    return {
        "type": "preference" if memory_type == "preference" else "token_stats",
        "data": {
            "key": key,
            "value": value,
            "memory_type": memory_type,
            "source_text": value,
            "created_at": int(time.time() * 1000),
            "updated_at": int(time.time() * 1000),
            "confidence": 1.0,
            "tags": [memory_type],
            "source": "benchmark",
            "layer": layer,
            "reinforcement_score": 0.0,
            "deleted": False,
        },
    }


def _start() -> StdioKernelClient:
    client = StdioKernelClient()
    client.start()
    return client


def run_ltm_benchmark() -> None:
    print("--- Layer 5: STM and Persistent LTM Benchmarks ---")
    client = _start()

    stm_keys = [f"session:benchmark:context:item_{index}" for index in range(100)]
    started = time.perf_counter()
    for key in stm_keys:
        client.store("stm", key, _record(key, "context", "stm", "context"))
    write_ms = (time.perf_counter() - started) * 1000 / len(stm_keys)

    recalls = 0
    started = time.perf_counter()
    for key in stm_keys:
        recalls += bool(client.get("stm", key).get("found"))
    read_ms = (time.perf_counter() - started) * 1000 / len(stm_keys)
    print(f"STM average write: {write_ms:.3f} ms")
    print(f"STM average read : {read_ms:.3f} ms ({recalls}/{len(stm_keys)} recalled)")

    preferences = [
        ("user:benchmark:preference:programming_language", "TypeScript"),
        ("user:benchmark:preference:theme", "Dark Mode"),
        ("user:benchmark:preference:testing", "Pytest"),
        ("user:benchmark:preference:framework", "React"),
    ]
    started = time.perf_counter()
    for key, value in preferences:
        client.store("ltm", key, _record(key, value, "ltm", "preference"))
    ltm_write_ms = (time.perf_counter() - started) * 1000 / len(preferences)

    # Add distractors to exercise lookup isolation without constructing raw wire data.
    for index in range(1_000):
        key = f"user:benchmark:fact:noise_{index}"
        client.store("ltm", key, _record(key, "noise", "ltm", "fact"))

    client.close(graceful=True)
    client = _start()
    try:
        successful = 0
        started = time.perf_counter()
        for key, expected in preferences:
            result = client.get("ltm", key)
            if result.get("found") and expected in str(result.get("payload")):
                successful += 1
        ltm_read_ms = (time.perf_counter() - started) * 1000 / len(preferences)
        print(f"LTM average write: {ltm_write_ms:.3f} ms")
        print(f"LTM restart read : {ltm_read_ms:.3f} ms")
        print(f"LTM persistence  : {successful}/{len(preferences)} preferences recalled")
        if successful != len(preferences):
            raise AssertionError("LTM did not survive a clean restart")
    finally:
        client.close(graceful=True)


if __name__ == "__main__":
    run_ltm_benchmark()
