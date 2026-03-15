# TST Memory System — Full System Specification

**Version:** 3.1  
**Status:** Implementation RFC

---

## 1. Overview

The TST Memory System is a local-first memory infrastructure for Small Language Models (SLMs) designed to operate efficiently on edge hardware.

The system separates intelligence into three independent components:

| Component | Responsibility |
|-----------|-----------------|
| **Models** | Generate outputs |
| **Memory** | Store structured evidence |
| **Kernel** | Control execution & safety |

This architecture allows small frozen models to accumulate experience over time without retraining.

### Core Idea

> **Models think.**  
> **Memory remembers.**  
> **Kernel decides.**

---

## 2. System Architecture

```
              User
               │
               ▼
           Chat SLM
               │
               ▼
          Interpreter
               │
               ▼
             Router
               │
               ▼
             Kernel
               │
        ┌──────┼──────┐
        │      │      │
        ▼      ▼      ▼
       STM    LTM    Tree
       Memory Memory Memory
        │
        ▼
       Worker SLM
        │
        ▼
      Response
```

---

## 3. Memory Hierarchy

| Layer | Purpose | Storage | Decay |
|-------|---------|---------|-------|
| **STM** | Session context | RAM | Fast |
| **LTM** | Persistent patterns | Disk | Slow |
| **Tree** | Structural relationships | Disk | None |

### Expected Query Distribution

| Layer | Share |
|-------|-------|
| **STM** | 60–70% |
| **LTM** | 20–30% |
| **Tree** | 5–10% |

---

## 4. Short-Term Memory (STM)

STM is the session cache optimized for extremely fast access.

### Structure

STM uses a ring buffer with hash index.

#### Capacity

- Default: 256 entries
- Range: 128–1024

#### Entry Layout

```
STMEntry {
  entry_id u32
  created_ts u64
  last_access_ts u64
  key_ref u32
  payload_ref u32
  reinforcement_score f32
  flags u8
}
```

### Index

Hash map:
```
key_hash → ring_slot
```

### Decay

Periodic decay:
```
reinforcement_score *= β
β ≈ 0.98
```

### Promotion

Entries may promote to LTM if:
```
reinforcement_score ≥ promotion_threshold
```

Default threshold:
- 10 accesses within window

### Performance

**Target:**
- Lookup: **< 1 ms**

---

## 5. Long-Term Memory (LTM)

LTM is implemented using an arena-backed Ternary Search Trie (TST).

This provides:
- Prefix compression
- Dynamic insert/delete
- Predictable memory usage
- Efficient serialization

---

## 6. TST Symbol Index

### Node Layout

```
Node {
  ch u8
  flags u8
  lo u32
  eq u32
  hi u32
  payload_idx u32
  next_leaf u32
}
```

**Node size:** 24 bytes

All references are arena indices, never pointers.

---

## 7. Payload Arena

Payloads store evidence and metadata.

Each payload begins with a common header.

### PayloadHeader

```
PayloadHeader {
  payload_type u16
  version u16
  created_ts u64
  last_access_ts u64
  access_count u32
}
```

---

## 8. Payload Schemas

### TokenStats

```
TokenStats {
  canonical_form string
  frequency u32
  decay_score f32
  preferred_tokenizer_origin enum optional
}
```

**Important:**
- `token_id` is **NOT** stored
- Token IDs are derived per model

### PhraseMeta

```
PhraseMeta {
  canonical_phrase string
  usage_count u32
  domain_mask u16
}
```

### ConceptAnchor

```
ConceptAnchor {
  concept_id u32
  related_tokens string[]
  strength f32
}
```

### StructurePattern

```
StructurePattern {
  pattern_id u32
  steps enum[]
  success_score f32
}
```

### Preference

```
Preference {
  key string
  value string
  weight f32
}
```

---

## 9. Tokenizer Alignment

Because different workers use different tokenizers:

- Memory stores **canonical strings**
- Kernel resolves token IDs per model

### Process

1. Retrieve canonical string
2. Tokenize using worker tokenizer
3. Cache token mapping

### Kernel Cache

Kernel maintains cache:
```
(model_family → string → token_ids)
```

**Cache size:** 50k entries default

---

## 10. Logit Biasing

Memory influences generation via logit bias.

### Formula

```
logits[token_id] += bias
```

Bias computed from payload evidence.

### Example

```
bias = log(1 + frequency) * decay_score
```

---

## 11. Bias Clamp

Bias range is model configurable.

### Default

```
[-1.5, +1.5]
```

But actual values depend on model.

### Configuration

```
model_config {
  bias_scale
  bias_clamp_min
  bias_clamp_max
}
```

---

## 12. Bias Calibration

Each model family requires calibration.

### Procedure

1. Collect logit statistics
2. Simulate bias injection
3. Measure effect on output distribution
4. Select safe clamp values

### Metrics Evaluated

- Perplexity
- Entropy
- Repetition
- Human coherence ratings

---

## 13. Memory Access API

Workers never access memory structures directly.

### Read API

```json
{
  "keys": ["token:global_warming"],
  "max_results": 32
}
```

**Response:**

```json
{
  "slices": [...]
}
```

### Write Proposal

Workers submit proposals.

```
Worker → Kernel → Memory
```

**Example:**

```json
{
  "op": "update",
  "key": "token:global_warming",
  "delta": {"frequency": 1}
}
```

Kernel validates before commit.

---

## 14. Conflict Resolution

### Precedence

```
STM > LTM
```

Session overrides always win temporarily.

### Promotion

STM updates propagate to LTM only if:

```
reinforcement ≥ threshold
OR user confirmation
```

### Aggregation

Counters merge:
```
freq = freq_a + freq_b
```

Preferences use weighted voting.

---

## 15. Tree Memory

Tree Memory stores structural relationships.

### Node Layout

```
TreeNode {
  node_id u64
  node_type enum
  name string
  parent u64
  children[]
  dependencies[]
}
```

### Node Types

- `PROJECT`
- `DIRECTORY`
- `FILE`
- `CLASS`
- `FUNCTION`
- `SYMBOL`
- `MODULE`

### Update Events

Tree updates on events:

- `file_added`
- `file_removed`
- `function_renamed`
- `dependency_changed`

### Query Interface

**Example:**

```
query_subgraph(node_id, depth)
```

**Example traversal:**

```
function_c → dependency → function_e
```

---

## 16. Persistence

Memory snapshots contain:

- Header
- Node arena
- Payload arena
- Free lists

Snapshots written atomically.

Optional WAL supports crash recovery.

---

## 17. Compaction

Compaction removes decayed entries.

### Steps

1. Traverse leaf list
2. Retain payloads above threshold
3. Rebuild trie
4. Write snapshot
5. Atomic swap

---

## 18. Concurrency

### Baseline Design

- Single RWLock
- Reads concurrent
- Writes serialized

**Edge hardware target:** 2–4 cores

---

## 19. Security

Kernel enforces:

- Prefix access controls
- Payload schema validation
- Write rate limits
- Poisoning detection

### Workers Cannot

- Modify memory directly
- Access arbitrary keys

---

## 20. Observability

All memory mutations logged:

- Timestamp
- Worker ID
- Payload ID
- Old value
- New value

### Metrics

- Latency
- Schema correctness
- Hallucination rate
- Escalation frequency

---

## 21. Worker Model Requirements

Workers must:

- Remain stateless
- Read memory slices only
- Produce structured outputs
- Support `insufficient_memory` response

---

## 22. Performance Targets

Edge device goals:

| Component | Target |
|-----------|--------|
| STM lookup | <1 ms |
| LTM lookup | <5 ms |
| Router inference | <150 ms |
| Worker generation | 2–4 s |

---

## 23. Memory Budget

Example system (~200k symbols):

| Component | Size |
|-----------|------|
| Node arena | 10–18 MB |
| Payload arena | 3–6 MB |
| Metadata | 2–4 MB |
| **Total** | **15–23 MB** |

---

## Summary

The TST Memory System provides a lightweight, auditable, persistent memory infrastructure for frozen SLMs. By separating models, memory, and kernel control, it enables edge-first deployment with deterministic, measurable performance.

**Key principles:**
- ✅ Models generate, memory stores, kernel controls
- ✅ Fast STM for session context, persistent LTM for long-term patterns
- ✅ Tree memory for structural relationships
- ✅ Canonical strings, per-model tokenization
- ✅ Logit bias for gentle influence without retraining
- ✅ Edge-first: ~15–23 MB for 200k symbols

---

**Document Version:** 3.1  
**Status:** Implementation RFC  
**Last Updated:** March 2026
