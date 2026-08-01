# Evaluation and regression checks

TST separates kernel latency from model inference latency. Pull-request CI
runs deterministic unit, contract and integration checks; heavyweight model
evaluations remain manual or scheduled.

## Pull-request checks

```bash
python -m pytest -m 'not integration'
python -m pytest -m protocol_contract
python -m pytest -m integration
cargo test --locked --manifest-path tst_memory/Cargo.toml --all-targets
```

The repository fixtures contain 300 labelled routing cases and 100 retrieval
cases. Deterministic scoring reports operation, layer and joint routing
accuracy; mutation safety; Recall@1/3; MRR; deletion leakage; and latency
percentiles. Dataset-size tests prevent accidental evaluation shrinkage.

```bash
python scripts/evaluate_routing.py
python scripts/evaluate_retrieval.py
python layer4_benchmarks.py test_project --with-kernel
python scripts/baseline.py
```

These commands enforce quality, safety, and latency gates by default.
`--report-only` is available when evidence must be captured without making the
process status an acceptance decision.

Repository-analysis tests verify source ranges, relative import and call
resolution, test relationships, unchanged-file skipping, deletion cleanup,
path traversal blocking and secret/symlink exclusions. Worker tests verify
strict JSON parsing, source line validation, symbol checks, deduplication and
low-confidence handling.

## Performance measurements

Measure the Rust kernel and model router independently. Record P50 and P95 for
each operation and include the input count, warm-up policy, hardware, OS,
Python/Rust versions and revision. Baseline capture is documented in
`baseline-v0.1.md`; new benchmark claims should link to the command and a
machine-readable result rather than relying on a prose-only number.

`scripts/baseline.py` records the revision, UTC timestamp, OS/architecture,
Python/Rust/Cargo versions, kernel binary path/hash/size, exact commands, Rust
test/stress outcomes, process RSS, startup/restart behavior, graph results, and
all v0.2 performance budgets. Use `--skip-rust-stress` only when explicitly
recording a lightweight diagnostic run.

The existing model comparison scripts require local weights or Ollama and are
not invoked by default CI.
