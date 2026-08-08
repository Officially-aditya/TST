# Getting Started with TST

This guide walks you through installing TST, building the kernel, and running your first memory operations.

## Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Rust stable toolchain** (edition 2024, tested on 1.93+)
- **macOS or Linux** (Windows via WSL2 is experimental)

Verify your toolchain:

```bash
python3 --version
cargo --version
rustc --version
```

## Installation

### 1. Clone and create virtual environment

```bash
git clone https://github.com/your-org/TST.git
cd TST
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Core install (no model dependencies)

```bash
python -m pip install -e .
```

This installs:
- Protocol and kernel client
- Memory planning and deterministic routing
- Repository scanner with Python AST parser
- CLI (`tst` command)

### 3. Build the Rust kernel

```bash
tst kernel build
```

This compiles `tst_memory/target/release/server` — the v1 protocol kernel binary.

### 4. Verify installation

```bash
tst doctor
```

Expected output shows Python, Cargo, kernel binary, and optional extras status.

## Quick Start: Your First Memory Operations

### Start a chat session (requires models extra)

```bash
# Install model runtime (one-time)
python -m pip install -e '.[models]'

# Start chat
tst chat
```

### Use the CLI for memory operations

```bash
# Store a preference
tst memory store --layer ltm --key "user:default:preference:language" --value "TypeScript"

# Retrieve it
tst memory get --layer ltm --key "user:default:preference:language"

# Search memory
tst memory search --layer ltm --query "language"
```

### Analyze a repository

```bash
# Build code graph for current directory
tst analyze .

# Query a specific symbol
tst analyze . --symbol run_route

# Get JSON output for scripting
tst analyze . --json
```

## Configuration

TST uses environment variables for runtime configuration. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TST_KERNEL_BIN` | `tst_memory/target/release/server` | Path to kernel binary |
| `TST_SNAPSHOT_PATH` | `.tst/ltm.snapshot` | LTM snapshot file location |
| `TST_STM_CAPACITY` | `256` | STM entry capacity |
| `TST_STM_HALF_LIFE_SECONDS` | `1800` | Decay half-life (30 min) |
| `TST_STM_PROMOTION_THRESHOLD` | `10.0` | Score threshold for STM→LTM promotion |
| `TST_SNAPSHOT_DEBOUNCE_MS` | `5000` | Debounce before snapshot write |

### Example: Custom STM for long sessions

```bash
export TST_STM_CAPACITY=1024
export TST_STM_HALF_LIFE_SECONDS=7200  # 2 hours
export TST_STM_PROMOTION_THRESHOLD=15.0
tst kernel build
tst chat
```

## Next Steps

- [Architecture Overview](architecture.md) — System design and boundaries
- [Protocol Specification](protocol.md) — v1 NDJSON protocol details
- [Memory Semantics](memory-semantics.md) — STM/LTM lifecycle and keys
- [Retrieval Pipeline](retrieval.md) — How queries find relevant memory
- [Code Graph](code-graph.md) — Repository analysis and symbol lookup
- [Structured Reviews](structured-reviews.md) — Validated code-review output
- [Evaluation](evaluation.md) — Regression checks and benchmarks
- [Troubleshooting](troubleshooting.md) — Common issues and diagnostics