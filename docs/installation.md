# Installation and command workflow

TST v0.2 supports Python 3.10 or newer and Rust edition 2024 on macOS and
Linux. Normal startup uses an already-built kernel binary; it does not invoke
Cargo implicitly.

## Core development install

From a clean checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
tst doctor
tst kernel build
```

`tst kernel build` runs an explicit release build and produces
`tst_memory/target/release/server`. Set `TST_KERNEL_BIN` to use a compatible
prebuilt binary elsewhere.

Kernel source is available in a source checkout and source distribution, but
the pure Python wheel does not pretend to contain a platform-independent Rust
executable. Wheel users must set `TST_KERNEL_BIN` to a compatible protocol-v1
server or use a distribution that bundles a binary under `tst/bin/<platform>`.
`tst doctor` reports which strategy is active.

The core install contains the protocol, kernel client, memory planning,
deterministic routing, repository scanner and Python AST parser. Large model
runtimes and native Tree-sitter grammars are optional:

```bash
# Syntax-aware JavaScript, TypeScript/TSX and Rust parsing
python -m pip install -e '.[analysis]'

# FastAPI wrapper
python -m pip install -e '.[router]'

# FunctionGemma/Qwen chat and model-backed routing
python -m pip install -e '.[models]'
```

Shell quoting around extras is recommended on zsh.

## Commands

```bash
tst doctor                  # show Python, Cargo, kernel and optional-extra status
tst doctor --json           # machine-readable diagnostics
tst kernel build            # explicitly compile the Rust release kernel
tst analyze path/to/repo    # safely build the local code graph
tst analyze . --symbol run  # retrieve one symbol and its immediate context
tst analyze . --json        # emit graph and indexing report as JSON
tst chat                    # start model-backed chat (models extra required)
```

Model IDs are configured with `TST_ROUTER_TIER1_MODEL` and
`TST_ROUTER_TIER2_MODEL`. Snapshot and STM runtime variables are documented in
[memory semantics](memory-semantics.md).

`tst analyze` never executes indexed source. Without the analysis extra,
Python still uses the standard-library AST and other languages use a
line-aware structural fallback. Install the analysis extra for the intended
Tree-sitter behavior on supported languages.

## Reproducible contributor checks

CI's lightweight Python toolchain is pinned in `requirements-dev.lock`:

```bash
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
python -m pytest -m 'not integration and not protocol_contract'
python -m pytest -m protocol_contract
ruff check tst router scripts tests layer4_benchmarks.py
ruff format --check tst router scripts tests layer4_benchmarks.py
mypy tst router

# Dedicated native-parser environment
python -m pip install -r requirements-analysis.lock
python -m pytest tests/analysis

cargo test --locked --manifest-path tst_memory/Cargo.toml --all-targets
cargo fmt --manifest-path tst_memory/Cargo.toml --all -- --check
cargo clippy --locked --manifest-path tst_memory/Cargo.toml --all-targets -- -D warnings
```

Model evaluations are intentionally excluded from pull-request CI because
they download weights and measure inference rather than kernel behavior.
The deterministic 300-case routing and 100-case retrieval gates do run in CI.
