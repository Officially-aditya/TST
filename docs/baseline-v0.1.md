# TST v0.1 baseline

This document records the reproducible behavior observed before the v0.2
architecture migration. It is intentionally factual: claims that depended on
locally cached model weights or dedicated benchmark hardware are not treated as
verified baseline results.

## Environment

- Baseline date: 2026-07-27
- Git revision: `8f4cc1c`
- Rust: `rustc 1.93.1` / `cargo 1.93.1`
- Python: CPython 3.14.6
- Platform: macOS (Apple host)

## Reproduced results

From the repository root:

```bash
cd tst_memory
cargo test --all-targets
```

The clean v0.1 tree passed 26 Rust tests: 22 library tests, two integration
tests and two layer-one tests. The compiler reported warnings but no failures.

The Python project had no package metadata, lockfile, or installed test runner,
so there was no single reproducible Python test command. The CLI built the Rust
binary on every startup and used a legacy line-prefix protocol, while the
FastAPI router attempted to contact an HTTP service at `127.0.0.1:3000` that
the repository did not provide.

## Captured protocol behavior

The v0.1 server printed `READY`, then accepted these line-prefixed commands:

- `WRITE <json>`
- `READ <json>`
- `TREE_INSERT <json>`
- `TREE_QUERY <json>`
- `TREE_LINK <json>`
- `TREE_CLEAR`

Unknown commands returned an unversioned `{"error":"Unknown command"}`
response. Requests had no IDs, timeouts, or structured error codes.

## Known semantic baseline

- LTM exact write/read worked in-process.
- STM used only a 32-bit key hash for identity.
- STM timestamps were initialized to zero by the kernel.
- Reads did not reliably update access metadata or trigger promotion.
- Snapshot save/load helpers passed isolated unit tests but were not called by
  the server during startup, mutation, or shutdown.
- CLI retrieval keys were truncated hashes of the complete input, so
  paraphrases could not find stored facts.
- Python, JavaScript/TypeScript, and Rust source scanning used regular
  expressions.

The v0.2 baseline runner writes current measurements as JSON so subsequent
changes can be compared without relying on this narrative record.
