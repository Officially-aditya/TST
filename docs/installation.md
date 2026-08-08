# Installation and Setup Guide

TST (Typed Structured Text) supports Python 3.10+ and Rust stable (edition 2024) on macOS and Linux.

## Platform Support Matrix

| OS | Architecture | Supported | Notes |
|----|--------------|-----------|-------|
| macOS | Apple Silicon (arm64) | Fully supported | Primary development platform |
| macOS | Intel (x86_64) | Supported | Requires Rust toolchain |
| Linux | x86_64 | Fully supported | Ubuntu 22.04+, Debian 12+, RHEL 9+ |
| Linux | AArch64 | Supported | Raspberry Pi 4+, ARM servers |
| Windows | WSL2 | Supported | Native Windows build experimental |

---

## Installation Scenarios

### Scenario A: Development Install (Source Checkout)

Recommended for developers modifying TST code or running tests.

```bash
# 1. Clone repository
git clone https://github.com/your-org/TST.git
cd TST

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Install core editable package
python -m pip install -e .

# 4. Verify system health
tst doctor

# 5. Build Rust release kernel
tst kernel build

# 6. Verify kernel build
ls -la tst_memory/target/release/server
```

---

### Scenario B: Full Installation (All Extras)

Includes JavaScript/TypeScript/Rust Tree-sitter parsing, FastAPI router, and chat models.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install with all extras (quote on zsh!)
python -m pip install -e '.[analysis,router,models]'

tst doctor
tst kernel build
```

**Extras breakdown**:

| Extra | Packages Installed | Purpose |
|-------|--------------------|---------|
| `[analysis]` | `tree-sitter`, `tree-sitter-languages` | Syntax-aware JS, TS, TSX, Rust parsing |
| `[router]` | `fastapi`, `uvicorn`, `httpx` | REST API service wrapper |
| `[models]` | `torch`, `transformers`, `accelerate` | FunctionGemma / Qwen chat model runtimes |

---

### Scenario C: Prebuilt Binary Distribution (Wheel Users)

For production deployment where building Rust with `cargo` is not desirable.

```bash
# Set environment variable pointing to prebuilt kernel binary
export TST_KERNEL_BIN=/opt/tst/bin/server

# Install Python package
pip install tst-memory

# Doctor verifies prebuilt binary
tst doctor
```

**Requirements for prebuilt binary**:
- Must implement TST Kernel Protocol Version 1 (NDJSON over STDIO)
- Must be executable by current user
- Must match host CPU architecture

---

## Post-Installation Verification

Run the full verification suite to ensure everything is operating correctly:

```bash
# 1. Check system diagnostics
tst doctor

# 2. Test code graph analysis
tst analyze . --json

# 3. Store and retrieve test memory
tst memory store --layer ltm --key "test:install:status" --value "ok"
tst memory get --layer ltm --key "test:install:status"

# 4. Clean up test memory
tst memory delete --layer ltm --key "test:install:status"
```

---

## Contributor Toolchain Setup

To run CI checks locally, install the pinned development dependencies:

```bash
# Install exact pinned CI dependencies
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .

# Install analysis dependencies for Tree-sitter tests
python -m pip install -r requirements-analysis.lock

# Run full test suite
python -m pytest -m 'not integration and not protocol_contract'
python -m pytest -m protocol_contract
python -m pytest -m integration
cargo test --locked --manifest-path tst_memory/Cargo.toml --all-targets

# Run linter and type-checking suite
ruff check tst router scripts tests layer4_benchmarks.py
ruff format --check tst router scripts tests layer4_benchmarks.py
mypy tst router
cargo fmt --manifest-path tst_memory/Cargo.toml --all -- --check
cargo clippy --locked --manifest-path tst_memory/Cargo.toml --all-targets -- -D warnings
```

---

## Environment Variable Reference

### Kernel Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `TST_KERNEL_BIN` | `tst_memory/target/release/server` | Path to compiled kernel binary |
| `TST_KERNEL_TIMEOUT` | `30.0` | Process request timeout in seconds |
| `TST_KERNEL_MAX_RESPONSE_BYTES` | `8388608` | Max response payload (8 MiB) |

### Memory Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `TST_SNAPSHOT_PATH` | `.tst/ltm.snapshot` | Path to persistent LTM snapshot |
| `TST_SNAPSHOT_DEBOUNCE_MS` | `5000` | Write debounce delay in milliseconds |
| `TST_STM_CAPACITY` | `256` | Maximum short-term memory entries |
| `TST_STM_HALF_LIFE_SECONDS` | `1800` | Score decay half-life in seconds |
| `TST_STM_PROMOTION_THRESHOLD` | `10.0` | Score required for STM → LTM promotion |

### Model Router

| Variable | Default | Description |
|----------|---------|-------------|
| `TST_ROUTER_TIER1_MODEL` | `google/gemma-2-2b-it` | Small router model ID |
| `TST_ROUTER_TIER2_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Medium router model ID |
| `TST_ROUTER_DEVICE` | `auto` | Device placement (`cpu`, `cuda`, `mps`, `auto`) |