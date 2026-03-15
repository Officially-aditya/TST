# TST Memory System

A three-tier structured memory kernel for sub-1B language models on edge hardware. Gives frozen SLMs persistent, structured memory without retraining — running entirely on-device.

---

## Architecture

```
User Input
    │
    ▼
Interpreter
    │
    ▼
Tiered Router
  Tier 1: FunctionGemma-270M
  Tier 2: Qwen3.5-0.8B (fallback)
    │
    ├── STM  — ring buffer       < 1 ms
    ├── LTM  — TST trie          < 5 ms
    └── Tree — DAG               < 20 ms
    │
    ▼
Worker SLM (Qwen3.5-0.8B)
    │
    ▼
Response
```

| Component | Language | Role |
|---|---|---|
| `tst_memory/` | Rust | Core kernel — STM, LTM, Tree, persistence, logit bias |
| `router/` | Python | Tiered router — FunctionGemma-270M → Qwen3.5-0.8B |
| `cli.py` | Python | Full end-to-end pipeline REPL |

### Memory Tiers

- **STM** — 256-slot ring buffer. Sub-millisecond read/write. Decays by score; promotes to LTM.
- **LTM** — Arena-backed Ternary Search Trie. 24-byte nodes, disk-persistent, atomic snapshots.
- **Tree** — DAG of project structure (File, Function, Class nodes with import edges). Import-graph-scoped subgraph queries.

---

## Results

| Metric | Target | Mac M2 (MPS) | Windows x86_64 (CPU) |
|---|---|---|---|
| STM write latency | < 1 ms | **0.009 ms** | **0.260 ms** |
| STM read latency | < 1 ms | **0.007 ms** | **0.158 ms** |
| LTM read latency | < 5 ms | **0.020 ms** | **0.154 ms** |
| Router accuracy | — | **4/4 (100%)** | **4/4 (100%)** |
| Router warm latency | — | ~1,350 ms | ~8,800 ms |
| Memory budget (200K symbols) | < 23 MB | **< 23 MB** | **< 23 MB** |
| Rust unit tests | — | 27/27 | 26/26 |
| Stress tests | — | 36/36 | 34/36 |

---

## How to Run

### Prerequisites

- Rust (2021 edition) — [rustup.rs](https://rustup.rs)
- Python 3.10+ with a virtualenv (project uses `gemma-env/`)
- Ollama with `qwen3.5:0.8b` pulled (for comparison evals only)

### 1. Build the Rust kernel

```bash
cd tst_memory
cargo build --release
```

### 2. Install Python dependencies

```bash
source gemma-env/bin/activate
pip install -r router/requirements.txt
```

### 3. Start the router server

```bash
source gemma-env/bin/activate
uvicorn router.server:app --port 8003
```

### 4. Run the CLI

```bash
source gemma-env/bin/activate
python cli.py
```

The CLI builds and starts the Rust kernel automatically, loads both models, then opens a REPL. Expect ~25–30 seconds on first run.

### Running Evals

```bash
# Router accuracy + latency
python router_eval.py

# Cross-file reasoning (TST vs Ollama)
python test_tree_eval.py

# Multi-file stress test (24 bugs, 10 files, 4 tiers)
python test_multifile_eval.py

# Kernel stress suite (36 tests)
cd tst_memory && cargo run --release --bin stress_bench
```

---

## Paper

[TST.pdf](TST.pdf) — full system description, ablation study, and benchmarks.
