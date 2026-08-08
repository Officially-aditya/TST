# TST Architecture v0.2

TST separates intent, memory planning, transport, and generation so a retrieval question cannot accidentally become a write merely because both use the same memory tier.

## High-Level Data Flow

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

## Component Boundaries

### 1. Action Router (`tst.routing`)

**Responsibility**: Classify user input into a canonical operation + memory layer.

**Returns**: Operation (store/get/update/delete/search), Layer (stm/ltm/tree), Source (deterministic/model), Confidence (0.0–1.0)

**Two routing modes**:
- **Deterministic** (default): Rule-based, zero model dependency, 100% reproducible
- **Model-backed** (optional): FunctionGemma/Qwen for ambiguous cases, requires `models` extra

**Key design**: Routing is pure classification — it never mutates memory. The operation/layer decision is explicit and auditable.

```python
# Example routing result
RoutingDecision(
    operation="memory.store",
    layer="ltm",
    source="deterministic",
    confidence=0.95,
    matched_pattern="preference:.*"
)
```

### 2. Memory Planner (`tst.memory.planner`)

**Responsibility**: Transform routed intent into canonical keys and structured records.

**Canonical key format**: `namespace:scope:category:identifier`

```text
user:default:preference:programming_language
session:abc123:context:service_name
project:tst:file:router%2Fserver.py
```

**Key normalization**: Segments are lowercased, whitespace collapsed, reserved characters percent-escaped. Reserved prefixes (`internal:`, `system:`) and traversal components (`..`) are rejected.

**Record structure**:
```python
MemoryRecord(
    key="user:default:preference:language",
    value="TypeScript",
    memory_type="preference",
    source_text="I prefer TypeScript for frontend work",
    created_at=1692345600.0,
    updated_at=1692345600.0,
    confidence=0.9,
    tags=["frontend", "language"],
    source="user_explicit",
    layer="ltm",
    reinforcement=2.0,
    deleted=False
)
```

### 3. Kernel Client (`tst.kernel`)

**Responsibility**: Sole Python owner of kernel process I/O. Manages subprocess lifecycle, request/response correlation, timeouts, and error handling.

**Transport**: STDIN/STDOUT NDJSON (one JSON object per line). No HTTP, no WebSockets.

**Key guarantees**:
- Request IDs echoed in responses for correlation
- Configurable timeout (default 30s) with process termination on timeout
- 8 MiB response size limit (configurable)
- stderr reserved for diagnostics only; stdout = `READY` + protocol lines
- Graceful shutdown flushes LTM before exit

### 4. Rust Kernel (`tst_memory`)

**Responsibility**: Owns all memory state, persistence, and protocol validation.

**Core subsystems**:
| Subsystem | Responsibility |
|-----------|----------------|
| STM | Session-scoped, decaying, promotable to LTM |
| LTM | Durable, snapshot-persisted, canonical-key upsert |
| Tree | Repository symbols/edges, rebuilt from source |
| Persistence | Atomic snapshots, checksums, corruption recovery |
| Protocol | v1 validation, unknown field rejection, metrics |

**Protocol v1**: Strict envelope with `protocol_version`, `request_id`, `operation`, `params`. Unknown fields rejected. 4 MiB request limit.

### 5. Repository Analysis (`tst.analysis`)

**Responsibility**: Build line-aware code graph from source without executing code.

**Parser strategy**:
| Language | Primary | Fallback |
|----------|---------|----------|
| Python | Tree-sitter | stdlib `ast` (always works) |
| JavaScript/JSX | Tree-sitter | Structural regex |
| TypeScript/TSX | Tree-sitter | Structural regex |
| Rust | Tree-sitter | Structural regex |

**Graph nodes**: Project, File, Symbol (function, class, struct, etc.)
**Graph edges**: `contains`, `defines`, `imports`, `calls`, `references`, `inherits`, `implements`, `tests`

**Incremental scanning**: SHA-256 per file; unchanged files skip parsing; changed files rebuild transactionally; deleted files remove subgraph.

**Safety**: Rooted scan (no traversal outside root), no symlink following, skips VCS/build/cache/secret/binary files, configurable byte/file limits.

### 6. Review Worker (`tst.worker`)

**Responsibility**: Generate structured code-review findings validated against indexed source.

**Output schema**: `CodeReviewOutput` containing `CodeIssue[]` — no prose allowed.

**Validation pipeline**:
1. Parse JSON (schema validation)
2. File exists in project, line range valid
3. Content hash matches current file (prevents stale findings)
4. Related symbols exist in named file (if code graph supplied)
5. Deduplication by identity
6. Confidence threshold filtering

**Model discipline**: `temperature=0`, `do_sample=false`, JSON schema injected. Raw output retained only in explicit debug mode.

## Runtime Dependencies

```
Core (always):
  pydantic >= 2.0

Optional extras:
  [analysis]  -> tree-sitter, tree-sitter-languages (JS/TS/Rust grammars)
  [router]    -> fastapi, uvicorn
  [models]    -> torch, transformers, accelerate, huggingface-hub
```

**Design principle**: Deterministic paths (routing, kernel, Python AST analysis) work with zero optional dependencies.

## Persistence Strategy

| Component | Persisted | Format |
|-----------|-----------|--------|
| LTM | Yes | `.tst/ltm.snapshot` (atomic, checksummed) |
| STM | No | In-memory only, rebuilds empty on restart |
| Tree | No | Rebuilt from source on each `tst analyze` |
| Protocol | N/A | NDJSON over STDIO |

**Snapshot guarantees**:
- Written to temp file → `fsync` → atomic rename
- Previous valid snapshot preserved as `ltm.snapshot.previous`
- Corrupt snapshot preserved as timestamped diagnostic
- Symlink targets rejected

## Non-Goals (v0.2)

| Category | Explicitly Not Included |
|----------|------------------------|
| Search | Vector/embedding search |
| Distribution | Multi-node, replication, clustering |
| Auth | Multi-user, RBAC, OAuth |
| UI | Desktop, web dashboard, TUI |
| Cloud | Managed service, hosted kernel |
| Execution | Auto-fix, code modification, CI/CD |
| Languages | Beyond Python/JS/TS/Rust |
| Tree persistence | Graph not saved to disk |
| Fine-tuning | Model training/RLHF |

**External escalation**: Only as a gated route decision when operator explicitly configures a provider. TST does not orchestrate the provider.

## Security Boundaries

- **No code execution**: Scanner never imports, evaluates, or runs analyzed code
- **No secret leakage**: Secret filenames (`.env`, `id_rsa`, `*.pem`, etc.) skipped by default
- **Path confinement**: All scans rooted; traversal attempts rejected
- **Protocol validation**: Unknown fields rejected; request/response size limits
- **Process isolation**: Kernel runs as separate process; Python cannot directly mutate kernel memory
- **Audit trail**: Request IDs correlated across router → planner → kernel → response