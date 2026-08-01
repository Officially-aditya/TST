# v0.2 architecture

TST separates intent, memory planning, transport and generation so a retrieval
question cannot accidentally become a write merely because both use the same
memory tier.

```text
input
  -> deterministic/model action router
  -> memory planner and canonical keys
  -> shared StdioKernelClient
  -> versioned NDJSON Rust kernel
  -> ranked memory or code-graph context
  -> local worker
  -> schema/source validation for reviews
```

## Boundaries

- `tst.routing` returns an operation and layer, with source and confidence.
- `tst.memory` builds canonical keys, records and bounded retrieval context.
- `tst.kernel` is the only Python owner of kernel process I/O. Both CLI and
  FastAPI callers use the same request/response envelopes.
- `tst.analysis` scans source under an explicit root, parses symbols and builds
  an in-memory repository graph. The graph is rebuilt from source.
- `tst.worker` defines strict code-review output and verifies each finding
  against indexed source before display.
- `tst_memory` owns STM lifecycle, persistent LTM, Tree operations, metrics and
  protocol validation.

The v1 protocol uses one JSON request and response per line. Request IDs are
echoed, and error diagnostics include an escaped request ID and code without
logging complete private request payloads.
STDIO is the only kernel transport for v0.2; no component assumes a Rust HTTP
service.

## Runtime and optional dependencies

Protocol and deterministic components have no model-runtime dependency.
Pydantic is the only core Python dependency. FastAPI, Torch/Transformers and
Tree-sitter grammar wheels are separate installation extras. This keeps
diagnostics, kernel operations and Python repository analysis usable on hosts
where model or native parser wheels are not installed.

The repository graph is deliberately not persisted in v0.2. Its schema is
still evolving and content-hash incremental parsing makes reconstruction
cheap for unchanged files.

## Deliberate non-goals

v0.2 does not add vector search, distributed memory, multi-user authentication,
a desktop or web UI, cloud orchestration, automatic fix execution, every
language, Tree persistence, or model fine-tuning. External escalation is only
a gated route decision when an operator explicitly configures a provider; TST
does not orchestrate that provider.
