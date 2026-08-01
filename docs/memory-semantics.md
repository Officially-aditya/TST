# Memory semantics

TST separates session-scoped STM from durable LTM. Tree Memory is rebuilt from
source and is never included in LTM snapshots.

## Canonical records

Python memory planning emits readable hierarchical keys such as:

```text
user:default:preference:programming_language
session:abc123:context:service_name
project:tst:file:router%2Fserver.py
```

Segments are normalized and percent escaped. Reserved internal prefixes,
non-canonical encoding, traversal components, empty segments, and oversized
keys are rejected. A structured record stores key, value, memory type, original
source text, creation/update timestamps, confidence, tags, source, layer,
reinforcement, and deletion state. The inner key and layer must match the
request envelope. Deletion uses `memory.delete`; clients cannot store a live
tombstone as an ordinary record.

## STM lifecycle

STM uses complete keys for equality, with a hash only as a collision-safe index.
Defaults are:

```text
capacity = 256
half_life_seconds = 1800
promotion_threshold = 10.0
read_reinforcement = 1.0
write_reinforcement = 2.0
expiry_score = 0.01
```

The corresponding environment variables are `TST_STM_CAPACITY`,
`TST_STM_HALF_LIFE_SECONDS`, `TST_STM_PROMOTION_THRESHOLD`,
`TST_STM_READ_REINFORCEMENT`, `TST_STM_WRITE_REINFORCEMENT`, and
`TST_STM_EXPIRY_SCORE`. Invalid, non-finite, or internally inconsistent values
prevent startup.

A successful read increments access count, updates last access, and adds read
reinforcement. An update adds the larger write reinforcement without pretending
to be a read. Scores decay by elapsed wall time using a configurable half-life.
Maintenance runs during access and on the server timer. Expired entries release
their payload slots. Ring holes are reused before a live entry is evicted.

At the promotion threshold, the complete key and synchronized payload move to
LTM, the record layer becomes `ltm`, STM storage is released, persistence is
marked dirty, and the promotion counter increments once. Canonical-key upsert
and STM removal make promotion idempotent.

## LTM persistence

The snapshot path defaults to `.tst/ltm.snapshot` and can be changed with
`TST_SNAPSHOT_PATH`. A mutation marks LTM dirty. A background maintenance tick
saves after `TST_SNAPSHOT_DEBOUNCE_MS` even when no further request arrives.
Graceful shutdown always flushes outstanding state.

Snapshots are bounded, checksummed, structurally validated, written to a
private temporary file, synced, and atomically renamed. Before replacement, a
validated primary becomes `ltm.snapshot.previous`. A corrupt primary is
preserved as a timestamped diagnostic file and recovery tries the previous
snapshot. Symlink snapshot files are rejected. Only LTM persists; STM starts
empty after restart.
