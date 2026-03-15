TST Memory System — Execution Log (Windows Machine Re-run)
============================================================
Machine: Windows 11 | GPU: NVIDIA GeForce MX130 (2GB VRAM) | CPU: x86_64
Python: 3.14.0 | Rust: 1.94.0 (freshly installed)
Date: 2026-03-14

---

## Phase 1: Rust Unit Tests (cargo test --release)

**Status: PASS**
**Result: 26/26 passed (0 failed)**

Breakdown:
- 22 unit tests (lib) — all passed
- 2 integration tests — all passed
- 2 layer1 tests — all passed

Warnings (non-blocking):
- unused import `std::mem::size_of` in tst.rs
- unused import `PayloadData` in kernel.rs
- unused variable `f2` in tree.rs

Note: context.md claimed 27 tests but actual count is 26. Likely a count discrepancy
from the original Mac session.

---

## Phase 2: Stress Benchmarks (stress_bench binary)

**Status: 34/36 PASS, 2 FAIL**

### Passing Tests (34):
- [2.1] TST Node Density Scaling: 1k through 1M — all 7 checkpoints PASS
  - 200k: lookup_p95=2,200ns, mem=22.42MB (spec: <100us, <25MB)
  - 1M: lookup_p95=2,700ns, mem=105MB (spec: <1ms, <120MB)
- [2.2] Pathological Key Distributions: all 4 PASS (worst: monotonic 1.33x)
- [2.3] Arena Fragmentation Under Churn: PASS (0.95x degradation after 1M ops)
- [2.5] Decay Model Boundary: all 5 PASS (exact formula match)
- [3.1] STM Ring Buffer Saturation: PASS (hit_rate=100%, p99=600ns)
- [3.2] STM-to-LTM Promotion: PASS (20/20 promoted, 20/20 verified)
- [3.3] STM Concurrent Access: PASS (0 errors over 5s)
- [4.2] ACL Enforcement: PASS (100% correctness)
- [4.3] WAL Crash Recovery: PASS (100/100 recovered, corruption handled gracefully)
- [5.1] Tree Scaling small/medium: PASS
- [5.2] Cyclic Dependency Handling: PASS (4,200ns, 3 nodes, no infinite loop)
- [6.1] Sustained Session 500 turns: PASS (late_p95=1,800ns, degradation=0.30x)
- [6.2] Rapid Task Switching: PASS (p50=700ns, p99=6,400ns, coherence=100%)
- [7.1] Adversarial Inputs: PASS (10/10, zero crashes)
- [8.1] Logit Bias Effect: PASS (all monotonicity/NaN/clamp checks)
- [8.2] Bias Clamp Safety: all 4 ranges PASS
- [9.1] Cross-Session Persistence: PASS (50/50 recall, 14,283B snapshot)

### Failing Tests (2):
- [5.1] tree_scaling_large: build=21ms query=15ms — **query 15ms > 5ms threshold**
- [5.1] tree_scaling_monorepo: build=106ms query=79ms — **query 79ms > 20ms threshold**

Root cause: Thresholds were calibrated for Apple M2. Windows x86_64 is slower on
tree DAG traversal (22,001 and 110,001 nodes). NOT a correctness bug — purely
hardware-performance gap.

---

## Phase 3: Kernel Latency Test (latency_test.py)

**Status: PASS**

- Write: 7,104 req/s | Avg 0.141 ms (spec: <1ms)
- Read:  7,390 req/s | Avg 0.135 ms (spec: <5ms)

Note: Significantly lower throughput than M2 (106k write, 137k read) due to
Windows STDIO subprocess overhead. Absolute latencies still well within spec.

---

## Phase 4: Layer 4 RAM Budget (layer4_benchmarks.py)

**Status: PASS**

- 20 mock coding tasks processed
- Sub-25MB RAM boundary: PASS

---

## Phase 5: Layer 5 STM/LTM Benchmarks (layer5_memory_benchmarks.py)

**Status: ALL PASS**

Required fix: Changed binary path from `./target/release/server` to absolute path
with `.exe` extension for Windows compatibility.

- STM Write: 0.260 ms avg — PASS (sub-1ms)
- STM Read: 0.158 ms avg — PASS
- STM recall accuracy: 100/100 — PASS (perfect hash eviction)
- LTM recall accuracy: 4/4 — PASS (perfect preference persistence through 1,000 noise injections)
- LTM avg read: 0.154 ms — PASS (sub-5ms)

---

## Phase 6: Layer 2 Baseline Tests (layer2_tests.py)

**Status: 1/4 PASS**

Required fixes:
- Changed `AutoModelForCausalLM` → `AutoModelForImageTextToText` (Qwen3.5-0.8B is multimodal)
- Changed `torch_dtype` → `dtype` (transformers 5.3.0 deprecation)
- Fixed device detection for Windows (no MPS; added CUDA/CPU fallback)

Results (CPU inference):
- Test 1: "What did we just discuss?" → Expected: STM, Got: STM — **PASS**
- Test 2: "User always prefers TypeScript" → Expected: LTM, Got: STM — **FAIL**
- Test 3: "Fix syntax error line 53" → Expected: Tree, Got: STM — **FAIL**
- Test 4: "What is the weather in Tokyo?" → Expected: Cloud, Got: STM — **FAIL**

Analysis: The old Layer 2 baseline approach (Qwen-only with manual chat template
+ regex extraction) routes everything to STM. This is the same behavior observed
on the original Mac session — the baseline deliberately demonstrates the limitations
of a single-model regex approach, which the new tiered router (Phase 7) is designed
to beat.

---

## Phase 7: Router Eval (router_eval.py)

**Status: COMPLETE**

Required fixes:
- Changed `AutoModelForCausalLM` → `AutoModelForImageTextToText` for Qwen3.5 (same as Phase 6)
- Fixed `get_device()` for Windows (no MPS; added CUDA/CPU fallback)
- Fixed Unicode box-drawing chars in `banner()` for Windows cp1252 console encoding
- FunctionGemma still uses `AutoModelForCausalLM` (correct — it's a pure causal LM)

### Baseline (Layer 2 approach — Qwen-only, regex extraction):
- Accuracy: **1/4 (25%)**
- Avg latency: **367,392 ms** (CPU)
- Test 1 (STM): PASS — Test 2 (LTM): FAIL — Test 3 (Tree): FAIL — Test 4 (Cloud): FAIL

### New Router (Tiered FunctionGemma-270M → Qwen3.5-0.8B fallback):
- Accuracy: **4/4 (100%)**
- Avg latency: **15,625 ms** (CPU)
- All 4 queries resolved at Tier 1 (FunctionGemma) — no Qwen fallback needed
- Test 1 (STM): PASS via FG — 34,463 ms
- Test 2 (LTM): PASS via FG — 8,839 ms
- Test 3 (Tree): PASS via FG — 10,449 ms
- Test 4 (Cloud): PASS via FG — 8,752 ms

### Delta Summary:
- Accuracy: 1/4 → 4/4 (+3 correct)
- Avg latency: 367,392 ms → 15,625 ms (-351,767 ms / ~23.5x faster)

Note: CPU latencies are much higher than the original Mac M2 session (where FG
resolved in ~50-150ms). However, the relative improvement ratio and 100% accuracy
are consistent across both machines.

---

## Final Summary

| Test Suite              | Result           | Notes                                 |
|-------------------------|------------------|---------------------------------------|
| Rust Unit Tests         | 26/26 PASS       |                                       |
| Stress Benchmarks       | 34/36 PASS       | 2 tree query latency fails (HW)      |
| Kernel Latency          | PASS             | 0.141ms write, 0.135ms read          |
| Layer 4 RAM Budget      | PASS             | Sub-25MB                              |
| Layer 5 STM/LTM         | ALL PASS         | 100/100 STM, 4/4 LTM                 |
| Layer 2 Baseline        | 1/4 (25%)        | Expected — baseline deliberately weak |
| Router Eval (New)       | **4/4 (100%)**   | FG Tier-1 resolved all, 23.5x faster |

### Windows Compatibility Fixes Applied:
1. `layer5_memory_benchmarks.py`: Absolute paths + `.exe` extension for kernel binary
2. `layer2_tests.py`: `AutoModelForImageTextToText` for Qwen3.5, `dtype` kwarg, device detection
3. `router_eval.py`: Same model class fix, device detection, Unicode encoding fix
4. All fixes are backward-compatible (platform detection via `platform.system()`)

### Known Platform Gaps (Windows vs Mac M2):
- Stress bench tree_scaling_large/monorepo: query latency exceeds M2-calibrated thresholds
- Kernel throughput: ~7k req/s vs ~120k req/s (Windows STDIO overhead)
- ML inference: ~23x slower on CPU vs M2 Apple Silicon — but relative accuracy identical
