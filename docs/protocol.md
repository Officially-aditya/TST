# Kernel protocol v1

TST uses one newline-delimited JSON request and response per line over the Rust
kernel's standard input and output. `StdioKernelClient` is the only Python
transport owner. The Rust process does not expose an HTTP endpoint.

## Envelope

```json
{
  "protocol_version": 1,
  "request_id": "request-42",
  "operation": "memory.get",
  "params": {
    "layer": "ltm",
    "key": "user:default:preference:editor"
  }
}
```

Every response echoes `request_id`, includes `ok`, and has exactly one of
`result` or `error`. Errors have a stable code and message. `metrics.kernel_ms`
measures kernel dispatch time; caller-observed latency additionally includes
serialization and process transport.

Unknown envelope fields and operation-specific parameter fields are rejected.
Requests are limited to 4 MiB. The Python client applies a configurable timeout
and an 8 MiB default response limit. A timeout or malformed/mismatched response
terminates that process because the stream can no longer be correlated safely.

## Operations

- `kernel.ping`, `kernel.status`, `kernel.shutdown`
- `memory.store`, `memory.get`, `memory.update`, `memory.delete`, `memory.search`
- `tree.clear`, `tree.insert`, `tree.remove`, `tree.link`, `tree.unlink`
- `tree.query`, `tree.find`
- `persistence.save`, `persistence.status`

Memory operations accept only `stm` or `ltm`; Tree operations use typed nodes
and edges. Tree queries enforce depth, node-count, and estimated-token budgets.
The complete operation catalog is checked by
`tests/fixtures/protocol/operation-catalog.json`.

## Process lifecycle

The server recovers LTM before printing `READY`. Normal startup requires an
existing binary and never invokes Cargo. `kernel.shutdown` flushes dirty LTM,
returns its response, and exits. EOF also flushes. stderr is reserved for
diagnostics; stdout contains only `READY` and protocol responses.

Protocol v1 is strict. A wire-incompatible envelope, operation, parameter, or
response change requires incrementing `protocol_version` or adding a tested
compatibility path. Run `python -m pytest -m protocol_contract` after every
protocol change.
