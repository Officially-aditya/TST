# Evaluation and Regression Checks

TST separates kernel latency from model inference latency. Pull-request CI runs deterministic unit, contract and integration checks; heavyweight model evaluations remain manual or scheduled.

## Pull-Request Checks

### Fast CI Pipeline

```bash
# 1. Python unit tests (no integration, no protocol contract)
python -m pytest -m 'not integration and not protocol_contract'

# 2. Protocol contract tests (fixture validation)
python -m pytest -m protocol_contract

# 3. Integration tests (requires kernel binary)
python -m pytest -m integration

# 4. Rust tests (all targets)
cargo test --locked --manifest-path tst_memory/Cargo.toml --all-targets
```

### Linting & Type Checking

```bash
# Python
ruff check tst router scripts tests layer4_benchmarks.py
ruff format --check tst router scripts tests layer4_benchmarks.py
mypy tst router

# Rust
cargo fmt --manifest-path tst_memory/Cargo.toml --all -- --check
cargo clippy --locked --manifest-path tst_memory/Cargo.toml --all-targets -- -D warnings
```

### Dedicated Parser Environment

```bash
# Tree-sitter grammars require separate lock file
python -m pip install -r requirements-analysis.lock
python -m pytest tests/analysis
```

---

## Test Categories

### Unit Tests (`-m 'not integration and not protocol_contract'`)

- Router decision logic (deterministic + model)
- Memory key normalization and planning
- Retrieval pipeline stages
- Protocol envelope validation
- Analysis parsers (Python AST, structural fallbacks)
- Graph query and resolution
- Review worker/validator

### Protocol Contract Tests (`-m protocol_contract`)

- Operation catalog completeness (`tests/fixtures/protocol/operation-catalog.json`)
- Envelope validation (unknown fields rejected)
- Request/response schema compliance
- Error code stability
- All 29 v1 operations tested

### Integration Tests (`-m integration`)

- Full kernel lifecycle (startup → READY → requests → shutdown)
- STM/LTM/Tree operations end-to-end
- Persistence save/load/recovery
- CLI commands (`tst doctor`, `tst analyze`, `tst memory`)
- FastAPI router endpoints (if `[router]` extra)
- Cross-component: router → planner → kernel → retrieval

### Rust Tests

```bash
# Unit tests in each module
cargo test --locked --manifest-path tst_memory/Cargo.toml --lib

# Integration tests (kernel process)
cargo test --locked --manifest-path tst_memory/Cargo.toml --test integration

# Stress/benchmark tests
cargo test --locked --manifest-path tst_memory/Cargo.toml --test stress
cargo test --locked --manifest-path tst_memory/Cargo.toml --test stress_bench
```

---

## Evaluation Datasets

### Routing Evaluation (300 cases)

`tst.evaluation.datasets.ROUTING_CASES` covers:

| Category | Count | Description |
|----------|-------|-------------|
| Explicit store | 50 | "Remember X", "Store Y" |
| Explicit retrieve | 50 | "What is X?", "Get Y" |
| Implicit preference | 40 | "I like TypeScript" → store |
| Contextual follow-up | 30 | "And for backend?" → retrieve |
| Update/delete | 30 | "Change X to Y", "Forget X" |
| Scope detection | 30 | Session vs persistent |
| Ambiguous/negative | 40 | "Hello", "Thanks" → no-op |
| Layer routing | 30 | STM vs LTM vs Tree |

**Metrics**: Operation accuracy, Layer accuracy, Joint accuracy, Latency P50/P95

### Retrieval Evaluation (100 cases)

`tst.evaluation.datasets.RETRIEVAL_CASES` covers:

| Category | Count | Description |
|----------|-------|-------------|
| Exact wording | 20 | Verbatim match |
| Paraphrase | 20 | Semantic equivalent |
| Pronoun follow-up | 15 | Context resolution |
| Conflicting prefs | 15 | Latest wins |
| Deletion handling | 10 | Deleted suppressed |
| Scope (temp/persistent) | 10 | STM vs LTM |
| Dense distractors | 10 | Many similar keys |

**Metrics**: Recall@1, Recall@3, MRR, Wrong-memory rate, Deleted leakage, Latency P50/P95

---

## Running Evaluations

### Routing

```bash
# Full evaluation with gates
python scripts/evaluate_routing.py

# Report only
python scripts/evaluate_routing.py --report-only --output routing-results.json

# With model router (requires [models] extra)
python scripts/evaluate_routing.py --router model
```

**Gates** (must pass):
- Joint operation+layer accuracy ≥ 95%
- P95 latency ≤ 5 ms
- Dataset size = 300 (prevents shrinkage)

### Retrieval

```bash
# Full evaluation with gates
python scripts/evaluate_retrieval.py

# Report only
python scripts/evaluate_retrieval.py --report-only --output retrieval-results.json
```

**Gates** (must pass):
- Recall@1 ≥ 95%
- Recall@3 ≥ 98%
- MRR ≥ 0.95
- Wrong-memory rate = 0%
- Deleted leakage = 0%
- P95 latency ≤ 5 ms
- Dataset size = 100

### Kernel Benchmarks

```bash
# Layer 4 benchmarks (requires test project)
python layer4_benchmarks.py test_project --with-kernel
```

Measures:
- STM exact read
- LTM exact read
- Lexical search
- Tree symbol lookup
- Tree subgraph query
- Snapshot save
- Protocol overhead
- Unchanged-file check
- Kernel startup/restart
- RSS memory

### Baseline Capture

```bash
# Full baseline (used for release benchmarks)
python scripts/baseline.py --kernel-bin tst_memory/target/release/server

# Lightweight (skip Rust stress)
python scripts/baseline.py --skip-rust-stress
```

**Captures**:
- Revision, UTC timestamp, OS/arch
- Python/Rust/Cargo versions
- Kernel binary path/hash/size
- Exact commands run
- Rust test/stress outcomes
- Process RSS (max)
- Startup/restart latency
- Graph query results
- All v0.2 performance budgets

---

## Performance Measurement Methodology

### Kernel Latency vs Model Inference

**Critical distinction**: Kernel latency = Rust kernel dispatch only. Model inference = Python router + model forward pass.

| Component | Measured By | Typical |
|-----------|-------------|---------|
| Kernel dispatch | `metrics.kernel_ms` in protocol response | < 1 ms |
| Serialization | Python client timing | ~0.1 ms |
| Process transport | Python client timing | ~0.2 ms |
| Router (deterministic) | `scripts/evaluate_routing.py` | ~0.02 ms |
| Router (model) | `scripts/evaluate_routing.py --router model` | ~50–200 ms |
| Retrieval | `scripts/evaluate_retrieval.py` | ~0.3 ms |

**Reporting rule**: Always separate kernel latency from model latency. Never conflate.

### Benchmark Environment

Document for every measurement:
- Hardware (CPU, RAM, disk type)
- OS + version
- Python version
- Rust/Cargo version
- Kernel binary: path, SHA-256, size
- Git revision
- Warm-up policy (e.g., 100 warm-up requests)
- Input count per operation
- P50 and P95 for each operation

### Reproducibility

```bash
# Exact reproduction commands from README v0.2 benchmark
tst kernel build
python scripts/evaluate_routing.py
python scripts/evaluate_retrieval.py
python layer4_benchmarks.py test_project --with-kernel
python scripts/baseline.py --kernel-bin tst_memory/target/release/server
```

---

## Model Evaluations (Excluded from CI)

### Why Excluded

- Require downloading model weights (GBs)
- Measure inference latency, not kernel behavior
- Non-deterministic (sampling, quantization variance)
- Hardware-dependent (GPU/MPS/CPU)
- Time-consuming (minutes vs seconds)

### When to Run

- Manual release validation
- Scheduled nightly (if infrastructure exists)
- Model comparison experiments
- Before upgrading model dependencies

### Model Comparison Scripts

```bash
# Requires local weights or Ollama
python scripts/model_comparison.py --models functiongemma,qwen --cases 300
```

---

## Adding New Tests

### Unit Test

```python
# tests/unit/test_new_feature.py
import pytest
from tst.new_feature import NewFeature

def test_new_feature_behavior():
    feature = NewFeature()
    assert feature.do_thing() == expected
```

### Integration Test

```python
# tests/integration/test_new_integration.py
import pytest
from tst.kernel.client import StdioKernelClient

@pytest.mark.integration
async def test_new_kernel_operation(kernel_client: StdioKernelClient):
    result = await kernel_client.request("new.operation", {...})
    assert result.ok
```

### Protocol Contract Test

Add to `tests/fixtures/protocol/operation-catalog.json`:
```json
{
  "operation": "new.operation",
  "request": { "protocol_version": 1, "request_id": "test", "operation": "new.operation", "params": {...} },
  "response_ok": { "protocol_version": 1, "request_id": "test", "ok": true, "result": {...} },
  "response_err": { "protocol_version": 1, "request_id": "test", "ok": false, "error": {"code": "INVALID_PARAMS", ...} }
}
```

Run: `python -m pytest -m protocol_contract`

---

## CI Configuration

### GitHub Actions (Example)

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Setup Python
        uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Install deps
        run: |
          python -m pip install -r requirements-dev.lock
          python -m pip install --no-deps -e .
      - name: Python tests
        run: |
          python -m pytest -m 'not integration and not protocol_contract'
          python -m pytest -m protocol_contract
      - name: Rust tests
        run: cargo test --locked --manifest-path tst_memory/Cargo.toml --all-targets
      - name: Lint
        run: |
          ruff check tst router scripts tests layer4_benchmarks.py
          ruff format --check tst router scripts tests layer4_benchmarks.py
          mypy tst router
          cargo fmt --manifest-path tst_memory/Cargo.toml --all -- --check
          cargo clippy --locked --manifest-path tst_memory/Cargo.toml --all-targets -- -D warnings
```

---

## Gate Enforcement

**All gates enforced by default** in evaluation scripts. Use `--report-only` to capture evidence without failing CI.

**Adding a new gate**:
1. Define metric + threshold in evaluation script
2. Add to `--report-only` JSON output
3. Document in this file
4. Update CI if gate should block merge