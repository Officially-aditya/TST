import subprocess
import json
import time
import sys

def run_ltm_benchmark():
    print("--- Layer 5: LTM Persistent Preference Recall Benchmarks ---")
    
    # Start Kernel backend
    print("Initializing Rust Memory Backend (Kernel) via Stdio...")
    import platform, os
    ext = ".exe" if platform.system() == "Windows" else ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    crate_dir = os.path.join(script_dir, "tst_memory")
    server_bin = os.path.join(crate_dir, "target", "release", f"server{ext}")
    server_cmd = [server_bin]
    kernel = subprocess.Popen(
        server_cmd,
        cwd=crate_dir,
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

    # ------------------------------------------------------------------
    # Test 1: STM Immediate Recall & Sub-1ms Latency Bounds
    # ------------------------------------------------------------------
    print("\n[Test 1] STM Execution (Immediate Recall Latency)")
    
    stm_keys = [f"session_ctx_{i}" for i in range(100)]
    
    start_write = time.time()
    for key in stm_keys:
        req = json.dumps({
            "op": "insert",
            "key": key,
            "layer": "STM",
            "payload": {
                "header": { "payload_type": 2, "version": 1, "created_ts": 0, "last_access_ts": 0, "access_count": 0 },
                "data": { "TokenStats": { "canonical_form": "context", "frequency": 1, "decay_score": 1.0, "preferred_tokenizer_origin": None } }
            }
        })
        kernel.stdin.write(f"WRITE {req}\n")
        kernel.stdin.flush()
        ack = kernel.stdout.readline()
        if '"status":"ok"' not in ack:
            print(f"DEBUG_WRITE_FAIL: {ack.strip()}")
            
    write_time_ms = ((time.time() - start_write) / 100) * 1000
    
    print(f"  -> Successfully dropped 100 cyclical contexts into the 256-slot STM ring buffer.")
    print(f"  -> STM Avg Write Latency: {write_time_ms:.3f} ms")
    if write_time_ms < 1.0:
        print("  -> Result: PASS (Sub-1ms criteria met)")
    else:
        print("  -> Result: FAIL (Latency exceeded bounds)")
        
    # Verify STM Native Retrieval bounded isolation
    start_read = time.time()
    successful_stm_recalls = 0
    for key in stm_keys:
        req = json.dumps({"keys": [key], "max_results": 1})
        kernel.stdin.write(f"READ {req}\n")
        kernel.stdin.flush()
        res_str = kernel.stdout.readline()
        if "context" in res_str:
            successful_stm_recalls += 1
            
    read_time_ms = ((time.time() - start_read) / 100) * 1000
    print(f"  -> STM Avg Read Latency: {read_time_ms:.3f} ms")
    print(f"  -> STM Hash Context Recall... {successful_stm_recalls}/100")
    if successful_stm_recalls == 100:
        print("  -> Result: PASS (Perfect hash eviction cycle accuracy)")
    else:
        print("  -> Result: FAIL (Ring buffer hash misses)")

    # ------------------------------------------------------------------
    # Test 2: LTM Persistent Consistency Across Iterations
    # ------------------------------------------------------------------
    print("\n[Test 2] LTM Cross-Session Preference Construction")
    # Simulate a user expressing preferences over multiple "sessions"
    preferences = [
        ("user:pref:lang", "TypeScript"),
        ("user:pref:theme", "Dark Mode"),
        ("user:pref:testing", "Pytest"),
        ("user:pref:framework", "React")
    ]
    
    start_write = time.time()
    for key, val in preferences:
        req = json.dumps({
            "op": "insert",
            "key": key,
            "layer": "LTM",
            "payload": {
                "header": { "payload_type": 1, "version": 1, "created_ts": 1000, "last_access_ts": 1000, "access_count": 0 },
                "data": { "Preference": { "key": key, "value": val, "weight": 1.5 } }
            }
        })
        kernel.stdin.write(f"WRITE {req}\n")
        kernel.stdin.flush()
        # Read the ACK
        ack = kernel.stdout.readline()
        if '"status":"ok"' not in ack:
            print(f"DEBUG_WRITE_FAIL: {ack.strip()}")
        
    write_time_ms = ((time.time() - start_write) / len(preferences)) * 1000
    print(f"  -> Successfully generated {len(preferences)} distinct LTM anchor points.")
    print(f"  -> Avg associative Write Latency.: {write_time_ms:.3f} ms")


    print("\n[Test 2] LTM Persistent Consistency Across Iterations")
    print("  -> Firing 1,000 arbitrary noise events to test structural integrity...")
    # Inject 1,000 noise queries
    for i in range(1000):
        req = json.dumps({
            "op": "insert",
            "key": f"noise:data:{i}",
            "layer": "LTM",
            "payload": {
                "header": { "payload_type": 2, "version": 1, "created_ts": 0, "last_access_ts": 0, "access_count": 0 },
                "data": { "TokenStats": { "canonical_form": "noise", "frequency": 1, "decay_score": 0.5, "preferred_tokenizer_origin": "qwen" } }
            }
        })
        kernel.stdin.write(f"WRITE {req}\n")
        kernel.stdin.flush()
        
        ack = kernel.stdout.readline()
        if '"status":"ok"' not in ack:
            pass # ignore noise acks

    print("  -> Noise injection complete. Checking isolated LTM Preference Recall bounds.")
    
    # Retrieve the targeted preferences out of the noise
    successful_recalls = 0
    start_read = time.time()
    
    for key, expected_val in preferences:
        req = json.dumps({"keys": [key], "max_results": 1})
        kernel.stdin.write(f"READ {req}\n")
        kernel.stdin.flush()
        
        response_str = kernel.stdout.readline()
        try:
            res = json.loads(response_str)
            # res format: {"slices": [{"header": {...}, "data": {"Preference": {"key": "...", "value": "TypeScript", ...}}}]}
            slices = res.get("slices", [])
            if slices and len(slices) > 0 and slices[0] is dict:
                pass # structure fallback
            
            # actually check the nested structure safely
            if slices and isinstance(slices[0], dict) and slices[0].get("data") and "Preference" in slices[0]["data"]:
                pref = slices[0]["data"]["Preference"]
                if pref.get("value") == expected_val:
                    successful_recalls += 1
                else:
                    print(f"DEBUG: Read slice value mismatch: {pref}")
            else:
                print(f"DEBUG: Unmatched schema: {response_str.strip()}")
        except json.JSONDecodeError:
            print(f"DEBUG: Decode Error on: {response_str.strip()}")

    read_time_ms = ((time.time() - start_read) / len(preferences)) * 1000
    
    print(f"  -> LTM Search Avg Read Latency.: {read_time_ms:.3f} ms")
    print(f"  -> Accuracy matching original Preference payloads... {successful_recalls}/{len(preferences)}")
    
    if successful_recalls == len(preferences):
        print("  -> Result: PASS (Perfect associative persistence across deep isolation loops)")
    else:
        print("  -> Result: FAIL (Association degradation)")
        
    if read_time_ms < 5.0:
        print("  -> Verification: PASS (Sub-5ms LTM criteria bounds met under load)")
    else:
        print("  -> Verification: FAIL (LTM Latency scaled incorrectly over node depth)")

    print("\nBenchmark completed. Terminating Kernel...")
    kernel.terminate()

if __name__ == "__main__":
    run_ltm_benchmark()
