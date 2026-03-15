import subprocess
import json
import time
import sys

def test_rust_kernel_latency():
    print("Building TST Memory Kernel for Profiling...")
    subprocess.run(["cargo", "build", "--release", "--bin", "server"], cwd="./tst_memory", check=True)

    server_cmd = ["cargo", "run", "--release", "--bin", "server"]
    kernel = subprocess.Popen(
        server_cmd,
        cwd="./tst_memory",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1
    )

    ready = kernel.stdout.readline()
    if "READY" not in ready:
        print("Kernel failed to start.")
        sys.exit(1)

    print("\n--- Latency Benchmark ---")
    
    # Write Latency (Insert 10,000 items)
    num_items = 10000
    start_time = time.time()
    for i in range(num_items):
        req = json.dumps({
            "op": "insert",
            "key": f"test_key_{i}",
            "payload": {
                "header": {
                    "payload_type": 1,
                    "version": 1,
                    "created_ts": 0,
                    "last_access_ts": 0,
                    "access_count": 0
                },
                "data": {
                    "TokenStats": {
                        "canonical_form": f"data_{i}",
                        "frequency": 1,
                        "decay_score": 1.0,
                        "preferred_tokenizer_origin": None
                    }
                }
            }
        })
        kernel.stdin.write(f"WRITE {req}\n")
        kernel.stdin.flush()
        _ = kernel.stdout.readline()
        
    write_time = time.time() - start_time
    print(f"Write Throughput: {num_items / write_time:.2f} requests/sec")
    print(f"Avg Write Latency: {(write_time / num_items) * 1000:.3f} ms")

    # Read Latency (Read 10,000 items)
    start_time = time.time()
    for i in range(num_items):
        req = json.dumps({"keys": [f"test_key_{i}"], "max_results": 1})
        kernel.stdin.write(f"READ {req}\n")
        kernel.stdin.flush()
        _ = kernel.stdout.readline()

    read_time = time.time() - start_time
    print(f"Read Throughput: {num_items / read_time:.2f} requests/sec")
    print(f"Avg Read Latency: {(read_time / num_items) * 1000:.3f} ms")

    kernel.terminate()

if __name__ == "__main__":
    test_rust_kernel_latency()
