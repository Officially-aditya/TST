# Memory Semantics

TST separates session-scoped STM from durable LTM. Tree Memory is rebuilt from source and is never included in LTM snapshots.

## Canonical Records

### Key Format

Python memory planning emits readable hierarchical keys:

```text
user:default:preference:programming_language
session:abc123:context:service_name
project:tst:file:router%2Fserver.py
```

**Structure**: `namespace:scope:category:identifier`

| Segment | Description | Examples |
|---------|-------------|----------|
| `namespace` | Ownership domain | `user`, `session`, `project`, `system` |
| `scope` | Specific instance | `default`, `abc123` (session ID), `tst` (project name) |
| `category` | Semantic type | `preference`, `context`, `fact`, `file`, `symbol` |
| `identifier` | Specific name | `programming_language`, `service_name`, `router/server.py` |

### Normalization Rules

1. **Lowercase**: All segments lowercased
2. **Whitespace**: Collapsed to single underscore (`_`)
3. **Reserved chars**: Percent-encoded (`/` → `%2F`, `:` → `%3A`, `%` → `%25`)
4. **Empty segments**: Rejected
5. **Traversal**: `..` and `.` rejected
6. **Reserved prefixes**: `internal:`, `system:`, `kernel:` rejected for user keys
7. **Max length**: 256 chars per key, 64 chars per segment

### Record Structure

```rust
pub struct MemoryRecord {
    pub key: String,                    // Canonical key
    pub value: String,                  // Payload
    pub memory_type: String,            // "preference" | "fact" | "context" | "code" | "procedure"
    pub source_text: String,            // Original user text that created this
    pub created_at: f64,                // Unix timestamp
    pub updated_at: f64,                // Unix timestamp
    pub confidence: f32,                // 0.0–1.0
    pub tags: Vec<String>,              // Searchable tags
    pub source: String,                 // "user_explicit" | "model_inferred" | "system" | "promoted"
    pub layer: Layer,                   // Stm | Ltm | Tree
    pub reinforcement: f32,             // Current reinforcement score
    pub deleted: bool,                  // Tombstone flag
    pub access_count: u64,              // Read count
    pub last_access: f64,               // Last read timestamp
}
```

**Layer consistency**: The `layer` field must match the request envelope layer. Mismatch returns `LAYER_MISMATCH` error.

### Deletion Semantics

- `memory.delete` sets `deleted = true` (tombstone)
- Clients **cannot** store a live tombstone as an ordinary record
- Retrieval filters out deleted records (newer delete suppresses older live version)
- LTM tombstones persisted in snapshot; STM tombstones release slot immediately

---

## STM Lifecycle

STM is a bounded, decaying cache with automatic promotion to LTM.

### Default Configuration

| Parameter | Default | Env Var | Description |
|-----------|---------|---------|-------------|
| Capacity | 256 | `TST_STM_CAPACITY` | Max entries |
| Half-life | 1800s (30 min) | `TST_STM_HALF_LIFE_SECONDS` | Score decay half-life |
| Promotion threshold | 10.0 | `TST_STM_PROMOTION_THRESHOLD` | Score to promote to LTM |
| Read reinforcement | 1.0 | `TST_STM_READ_REINFORCEMENT` | Score added on read |
| Write reinforcement | 2.0 | `TST_STM_WRITE_REINFORCEMENT` | Score added on write/update |
| Expiry score | 0.01 | `TST_STM_EXPIRY_SCORE` | Score below which entry expires |

### Score Dynamics

**Reinforcement** (increases score):
- `memory.get` → `+read_reinforcement`
- `memory.store` → `+write_reinforcement`
- `memory.update` → `+write_reinforcement` (not read)

**Decay** (decreases score continuously):
```
score(t) = score_0 * 0.5^(elapsed_seconds / half_life_seconds)
```

**Maintenance** runs:
- On every access (read/write)
- On server timer (every 60s by default)
- During promotion check

### Eviction Policy

When at capacity and new entry arrives:
1. Remove expired entries (score < `expiry_score`)
2. Reuse ring holes from expired entries
3. If still full, evict lowest-score live entry

**Ring buffer**: STM uses a fixed-capacity ring. Holes from eviction/expiration are reused before overwriting live entries.

### Promotion to LTM

When an entry's score ≥ `promotion_threshold`:

1. **Complete key + payload** moved to LTM (canonical-key upsert)
2. Record `layer` becomes `ltm`
3. STM storage released (slot freed)
4. LTM persistence marked dirty
5. Promotion counter incremented (once per key)

**Idempotency**: Promotion uses canonical-key upsert + STM removal. Retrying promotion is safe.

### STM Configuration Validation

Invalid values prevent kernel startup:
- Non-finite numbers (NaN, infinity)
- Negative values where positive required
- `half_life_seconds` ≤ 0
- `capacity` < 1 or > 100000
- `promotion_threshold` < `expiry_score`
- `read_reinforcement` < 0 or `write_reinforcement` < 0

---

## LTM Persistence

### Snapshot File

- **Default path**: `.tst/ltm.snapshot` (relative to kernel working dir)
- **Configurable**: `TST_SNAPSHOT_PATH` environment variable
- **Format**: Bounded, checksummed, structurally validated JSON

### Write Trigger

LTM marked dirty on any mutation (store/update/delete/promotion). Write occurs when:
1. `persistence.save` request received, OR
2. Background maintenance tick after `TST_SNAPSHOT_DEBOUNCE_MS` (default 5000ms) with no further requests, OR
3. Graceful shutdown (`kernel.shutdown` or EOF)

### Atomic Write Procedure

```
1. Serialize LTM to JSON
2. Compute SHA-256 checksum
3. Write to temp file: .tst/ltm.snapshot.tmp.<pid>.<timestamp>
4. fsync temp file
5. Rename current snapshot → .tst/ltm.snapshot.previous (if exists)
6. Rename temp → .tst/ltm.snapshot (atomic)
7. Update metadata
```

### Recovery

On startup:
1. Read `.tst/ltm.snapshot`
2. Validate checksum
3. Validate structure (required fields, types, no duplicates)
4. If valid → load
5. If corrupt → preserve as `.tst/ltm.snapshot.corrupt.<timestamp>`
6. Try `.tst/ltm.snapshot.previous` (validate same way)
7. If both fail → start with empty LTM, log diagnostics

### Symlink Protection

- Snapshot path resolved with `canonicalize()`
- Symlink targets rejected at startup
- Prevents path traversal via symlink swap

### What Persists

| Component | Persisted? |
|-----------|------------|
| LTM records | Yes (all non-deleted + tombstones) |
| STM records | No (rebuilds empty) |
| Tree graph | No (rebuilt from source) |
| Promotion counters | Yes (in LTM record metadata) |
| STM ring state | No |

---

## Environment Variable Reference

### STM Runtime

```bash
export TST_STM_CAPACITY=512                    # default 256
export TST_STM_HALF_LIFE_SECONDS=3600          # default 1800 (30 min)
export TST_STM_PROMOTION_THRESHOLD=15.0        # default 10.0
export TST_STM_READ_REINFORCEMENT=1.5          # default 1.0
export TST_STM_WRITE_REINFORCEMENT=3.0         # default 2.0
export TST_STM_EXPIRY_SCORE=0.005              # default 0.01
```

### Persistence

```bash
export TST_SNAPSHOT_PATH=/data/tst/ltm.snapshot     # default .tst/ltm.snapshot
export TST_SNAPSHOT_DEBOUNCE_MS=10000               # default 5000
```

### Kernel

```bash
export TST_KERNEL_BIN=/opt/tst/bin/server           # default tst_memory/target/release/server
export RUST_LOG=tst_memory=debug                    # kernel logging
```

---

## Example: Memory Flow

```python
# User says: "Remember I use TypeScript for frontend"
# 1. Router classifies: memory.store, layer=ltm, confidence=0.95
# 2. Planner creates canonical key: user:default:preference:programming_language
# 3. Kernel receives store request
# 4. LTM: canonical-key upsert, mark dirty
# 5. Background tick (5s later): atomic snapshot write

# Later, user asks: "What language for frontend?"
# 1. Router classifies: memory.get, layer=ltm
# 2. Kernel exact lookup → returns record
# 3. STM not involved (layer=ltm)

# User says: "Actually I switched to Rust"
# 1. Router: memory.update, layer=ltm
# 2. Kernel: updates value, adds write_reinforcement, mark dirty

# Session-only context: "Call the service Atlas"
# 1. Router: memory.store, layer=stm (detected as temporary)
# 2. STM: adds with write_reinforcement=2.0
# 3. Read later: adds read_reinforcement=1.0
# 4. After 30 min half-life: score decays
# 5. If promoted (score≥10): moves to LTM
# 6. If expired (score<0.01): removed on maintenance
# 7. Restart: STM empty, context gone
```

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Store existing key (LTM) | Upsert (replace value, update metadata) |
| Store existing key (STM) | Upsert in ring, reset reinforcement |
| Update non-existent key | `KEY_NOT_FOUND` error |
| Delete non-existent key | Success (idempotent) |
| Get deleted key | Returns `null` (filtered) |
| Search finds deleted | Filtered out by deletion timestamp |
| Promote already-promoted key | Idempotent (LTM upsert + STM remove) |
| Snapshot during promotion | Promotion completes, included in next snapshot |
| Corrupt snapshot + no previous | Empty LTM, corrupt preserved for debugging |