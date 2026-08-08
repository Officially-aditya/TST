# Kernel Protocol v1

TST uses one newline-delimited JSON request and response per line over the Rust kernel's standard input and output. `StdioKernelClient` is the only Python transport owner. The Rust process does not expose an HTTP endpoint.

## Envelope Format

### Request

```json
{
  "protocol_version": 1,
  "request_id": "req-42",
  "operation": "memory.get",
  "params": {
    "layer": "ltm",
    "key": "user:default:preference:editor"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `protocol_version` | integer | Yes | Must be `1` |
| `request_id` | string | Yes | Client-generated, echoed in response |
| `operation` | string | Yes | One of the operations below |
| `params` | object | Yes | Operation-specific parameters |

### Response (Success)

```json
{
  "protocol_version": 1,
  "request_id": "req-42",
  "ok": true,
  "result": {
    "key": "user:default:preference:editor",
    "value": "TypeScript",
    "memory_type": "preference",
    "layer": "ltm",
    "created_at": 1692345600.0,
    "updated_at": 1692345600.0,
    "confidence": 0.9,
    "tags": ["frontend"],
    "source": "user_explicit",
    "reinforcement": 2.0
  },
  "metrics": {
    "kernel_ms": 0.031
  }
}
```

### Response (Error)

```json
{
  "protocol_version": 1,
  "request_id": "req-42",
  "ok": false,
  "error": {
    "code": "KEY_NOT_FOUND",
    "message": "Key not found in layer ltm: user:default:preference:editor"
  },
  "metrics": {
    "kernel_ms": 0.012
  }
}
```

**Error codes**: `INVALID_ENVELOPE`, `UNKNOWN_OPERATION`, `INVALID_PARAMS`, `KEY_NOT_FOUND`, `KEY_INVALID`, `LAYER_MISMATCH`, `CAPACITY_EXCEEDED`, `PROMOTION_FAILED`, `PERSISTENCE_ERROR`, `INTERNAL_ERROR`

### Constraints

- **Unknown fields rejected** at envelope and params level (strict schema)
- **Request size limit**: 4 MiB
- **Response size limit**: 8 MiB (configurable in Python client)
- **Timeout**: Configurable (default 30s); timeout terminates kernel process
- **NDJSON**: Exactly one JSON object per line; no pretty-printing

---

## Operations Reference

### Kernel Management

#### `kernel.ping`
Health check. Returns `{ "status": "ok", "uptime_ms": 12345 }`.

#### `kernel.status`
Detailed status. Returns:
```json
{
  "version": "0.2.0",
  "protocol_version": 1,
  "uptime_ms": 12345,
  "stm": { "entries": 12, "capacity": 256, "promotions": 3 },
  "ltm": { "entries": 45, "dirty": false, "snapshot_path": ".tst/ltm.snapshot" },
  "tree": { "nodes": 1234, "edges": 5678 },
  "persistence": { "last_save_ms": 1692345600000, "save_count": 12 }
}
```

#### `kernel.shutdown`
Graceful shutdown. Flushes dirty LTM, returns response, exits process. EOF on stdin has same effect.

---

### Memory Operations (STM/LTM)

All memory operations accept `layer` parameter: `"stm"` or `"ltm"`.

#### `memory.store`
Store a new record or upsert existing key.

**Params**:
```json
{
  "layer": "ltm",
  "key": "user:default:preference:language",
  "value": "TypeScript",
  "memory_type": "preference",
  "source_text": "I prefer TypeScript for frontend",
  "confidence": 0.9,
  "tags": ["frontend", "language"],
  "source": "user_explicit"
}
```

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `layer` | string | Yes | — |
| `key` | string | Yes | — |
| `value` | string | Yes | — |
| `memory_type` | string | No | `"fact"` |
| `source_text` | string | No | `""` |
| `confidence` | float | No | `1.0` |
| `tags` | string[] | No | `[]` |
| `source` | string | No | `"unspecified"` |

**Result**: Stored record (see response format above)

**Behavior**:
- STM: Adds with `write_reinforcement` (default 2.0), may trigger promotion
- LTM: Canonical-key upsert (replaces existing), marks persistence dirty

#### `memory.get`
Exact key lookup.

**Params**:
```json
{ "layer": "ltm", "key": "user:default:preference:language" }
```

**Result**: Record or `null` if not found

**Side effect**: Adds `read_reinforcement` (default 1.0), updates access time/count

#### `memory.update`
Update value and/or metadata of existing key.

**Params**:
```json
{
  "layer": "ltm",
  "key": "user:default:preference:language",
  "value": "TypeScript (strict mode)",
  "confidence": 0.95,
  "tags": ["frontend", "language", "strict"]
}
```

| Param | Type | Required |
|-------|------|----------|
| `layer` | string | Yes |
| `key` | string | Yes |
| `value` | string | No |
| `confidence` | float | No |
| `tags` | string[] | No |

**Result**: Updated record

**Behavior**: Adds `write_reinforcement` (not `read_reinforcement`). Layer must match existing record.

#### `memory.delete`
Mark key as deleted (tombstone). Does not remove from LTM immediately.

**Params**:
```json
{ "layer": "ltm", "key": "user:default:preference:editor" }
```

**Result**: `{ "deleted": true, "key": "..." }`

**Behavior**: 
- STM: Removes entry immediately, releases slot
- LTM: Sets `deleted=true`, marks persistence dirty. Newer delete suppresses older live versions in retrieval.

#### `memory.search`
Lexical search with ranking pipeline.

**Params**:
```json
{
  "layer": "ltm",
  "query": "language preference",
  "limit": 10,
  "offset": 0,
  "min_score": 0.1
}
```

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `layer` | string | Yes | — |
| `query` | string | Yes | — |
| `limit` | integer | No | `10` |
| `offset` | integer | No | `0` |
| `min_score` | float | No | `0.0` |

**Result**:
```json
{
  "results": [
    { "record": {...}, "score": 0.87, "match_type": "lexical" },
    { "record": {...}, "score": 0.62, "match_type": "prefix" }
  ],
  "total_candidates": 45,
  "search_ms": 1.2
}
```

**Pipeline**: exact → prefix/category → lexical scoring → rerank → dedup → budget (see [Retrieval](retrieval.md))

---

### Tree Operations

Tree operations use typed nodes and edges. Keys are not used; operations reference nodes by `node_id`.

#### `tree.clear`
Remove all nodes and edges.

**Params**: `{}`

**Result**: `{ "cleared_nodes": 1234, "cleared_edges": 5678 }`

#### `tree.insert`
Insert or update a node.

**Params**:
```json
{
  "node_id": "project:tst:file:router/server.py:symbol:run_route",
  "node_type": "symbol",
  "label": "run_route",
  "file_path": "router/server.py",
  "line_start": 42,
  "line_end": 67,
  "signature": "async def run_route(req: Request) -> Response",
  "content_hash": "sha256:...",
  "metadata": { "kind": "function", "async": true }
}
```

**Result**: Inserted/updated node

#### `tree.remove`
Remove node and all connected edges.

**Params**: `{ "node_id": "..." }`

**Result**: `{ "removed_node": true, "removed_edges": 12 }`

#### `tree.link`
Create directed edge between nodes.

**Params**:
```json
{
  "source": "project:tst:file:router/server.py:symbol:run_route",
  "target": "project:tst:file:kernel/client.py:symbol:StdioKernelClient",
  "edge_type": "calls"
}
```

**Edge types**: `contains`, `defines`, `imports`, `calls`, `references`, `inherits`, `implements`, `tests`

#### `tree.unlink`
Remove specific edge.

**Params**:
```json
{ "source": "...", "target": "...", "edge_type": "calls" }
```

#### `tree.query`
Graph traversal from a starting node.

**Params**:
```json
{
  "start_node": "project:tst:file:router/server.py:symbol:run_route",
  "directions": ["callers", "callees", "imports"],
  "max_depth": 2,
  "max_nodes": 50,
  "token_budget": 2000
}
```

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `start_node` | string | Yes | — |
| `directions` | string[] | No | `["callers","callees","imports","types","tests","parents","children"]` |
| `max_depth` | integer | No | `3` |
| `max_nodes` | integer | No | `100` |
| `token_budget` | integer | No | `4000` |

**Result**:
```json
{
  "nodes": [...],
  "edges": [...],
  "truncated": false,
  "token_estimate": 1234
}
```

#### `tree.find`
Find nodes by label pattern.

**Params**:
```json
{ "pattern": "run_*", "node_types": ["symbol"], "limit": 20 }
```

---

### Persistence Operations

#### `persistence.save`
Force immediate LTM snapshot write.

**Params**: `{}`

**Result**: `{ "saved": true, "path": ".tst/ltm.snapshot", "entries": 45, "bytes": 12345, "duration_ms": 10.7 }`

#### `persistence.status`
Snapshot status.

**Params**: `{}`

**Result**:
```json
{
  "path": ".tst/ltm.snapshot",
  "exists": true,
  "size_bytes": 12345,
  "entries": 45,
  "last_modified": 1692345600.0,
  "checksum": "sha256:...",
  "previous_exists": true
}
```

---

## Process Lifecycle

### Startup

1. Kernel binary starts
2. Recovers LTM from snapshot (if exists)
3. Validates snapshot checksum and structure
4. Prints `READY` to stdout (single line, no JSON)
5. Accepts protocol requests on stdin

### Normal Operation

- Each request → single response (same `request_id`)
- `metrics.kernel_ms` measures kernel dispatch only
- Caller-observed latency = kernel_ms + serialization + process transport

### Shutdown

- `kernel.shutdown` request → flush LTM → respond → exit(0)
- EOF on stdin → flush LTM → exit(0)
- SIGTERM → flush LTM → exit(0)
- Crash/SIGKILL → next startup recovers from last valid snapshot

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Malformed JSON | Process terminates (stream correlation lost) |
| Unknown envelope field | Error response, process continues |
| Unknown operation | Error response, process continues |
| Invalid params | Error response, process continues |
| Request timeout | Python client kills process, restarts |
| Response size exceeded | Python client kills process, restarts |
| Kernel panic | Process exits, Python client restarts |

---

## Testing Protocol Changes

The complete operation catalog is defined in `tests/fixtures/protocol/operation-catalog.json` and validated by:

```bash
python -m pytest -m protocol_contract
```

**Protocol evolution rules**:
1. Wire-incompatible change → increment `protocol_version`
2. Compatible addition → add operation/param, update catalog, add tests
3. Run contract tests after every protocol change
4. Never assume HTTP transport — STDIO only for v0.2