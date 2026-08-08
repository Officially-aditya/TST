# Memory Retrieval

v0.2 intentionally uses no vector database. Retrieval is deterministic, inspectable, and inexpensive enough for edge hardware.

## Pipeline Overview

Production dispatch and evaluation use the same `RetrievalPipeline`:

```
1. Exact canonical-key lookup
2. Bounded prefix/category candidate lookup
3. Token-based lexical scoring with conservative synonym groups
4. Reinforcement, recency, and source-confidence reranking
5. Newest-version deduplication and deletion filtering
6. Item and character-budget context selection
```

Each stage narrows candidates; early stages are fast filters, later stages do precise ranking.

---

## Stage 1: Exact Lookup

**Input**: Canonical key (from planner or explicit `memory.get`)

**Action**: Direct hash-map lookup in requested layer (STM or LTM)

**Output**: Single record or `None`

**Latency**: < 0.05 ms P95

**Use case**: Explicit references ("my preference for language")

---

## Stage 2: Prefix/Category Candidate Expansion

**Input**: Query string, layer

**Action**: 
- Split query into tokens
- For each token, find keys with matching prefix (e.g., `user:default:preference:*`)
- Include category-matched keys (e.g., all `preference` type records)
- Bound: max 500 candidates per token, max 2000 total

**Output**: Candidate record set

**Latency**: < 0.5 ms P95

---

## Stage 3: Lexical Scoring

**Algorithm**: Token-based TF-IDF-like scoring with conservative synonym expansion

```python
def lexical_score(query: str, record: MemoryRecord) -> float:
    query_tokens = tokenize(query)
    record_tokens = tokenize(record.value + " " + record.key + " " + " ".join(record.tags))
    
    # TF component
    tf_scores = {t: record_tokens.count(t) / len(record_tokens) for t in query_tokens}
    
    # IDF component (precomputed per layer)
    idf_scores = {t: idf[t] for t in query_tokens if t in idf}
    
    # Synonym boost (conservative, curated groups)
    syn_boost = synonym_boost(query_tokens, record_tokens)
    
    return sum(tf_scores[t] * idf_scores.get(t, 1.0) for t in query_tokens) + syn_boost
```

**Synonym groups** (built-in, not learned):
```python
SYNONYM_GROUPS = [
    {"language", "lang", "programming language", "pl"},
    {"editor", "ide", "code editor"},
    {"frontend", "front-end", "client", "ui", "web"},
    {"backend", "back-end", "server", "api"},
    {"database", "db", "datastore", "storage"},
    {"test", "testing", "spec", "specs", "unit test"},
    {"config", "configuration", "settings", "prefs", "preferences"},
]
```

**Conservative design**: No learned embeddings, no external thesaurus. Groups are explicit and auditable.

**Output**: Candidates with `lexical_score` (0.0–1.0 normalized)

---

## Stage 4: Reranking

**Weighted composite score**:

```
final_score = 
    0.45 * lexical_similarity      # Stage 3 output
  + 0.20 * key_prefix_match        # 1.0 if query matches key prefix, else 0.0
  + 0.15 * reinforcement           # Normalized STM/LTM reinforcement (0–1)
  + 0.10 * recency                 # Exponential decay: exp(-hours_since_update / 168)
  + 0.10 * source_confidence       # record.confidence (0–1)
```

**Layer filtering**: Applied BEFORE ranking. Query specifies `stm` or `ltm`; cross-layer search not supported in single request.

**Normalization**: Each component scaled to 0–1 range before weighting.

---

## Stage 5: Deduplication & Deletion Filtering

**Newest-version deduplication**:
- Group by canonical key
- Keep only highest `updated_at` version
- Deleted records (tombstones) suppress older live versions of same key

**Deletion filtering**:
- Records with `deleted=true` excluded from results
- Tombstones retained in LTM for suppression logic only

**Output**: Deduplicated, deletion-filtered candidate list

---

## Stage 6: Context Selection

**Budget constraints** (configurable per request):

| Budget | Default | Description |
|--------|---------|-------------|
| `max_items` | 10 | Max records returned |
| `max_chars` | 4000 | Max total characters across all records |
| `max_tokens` | 1000 | Estimated token budget (chars/4) |

**Selection algorithm**:
1. Sort by `final_score` descending
2. Iterate, adding records while budgets not exceeded
3. If next record exceeds budget, stop (no partial records)

**Result format**:
```json
{
  "results": [
    { "record": {...}, "score": 0.87, "match_type": "lexical" },
    { "record": {...}, "score": 0.62, "match_type": "prefix" }
  ],
  "total_candidates": 45,
  "search_ms": 1.2,
  "budgets": { "max_items": 10, "max_chars": 4000, "used_chars": 2341 }
}
```

---

## Read-Only Guarantee

**Critical**: Exact retrieval and fallback search are **read-only**. A question is never stored as memory.

- `memory.get` → adds read reinforcement only
- `memory.search` → no mutation, no side effects
- No "query becomes memory" behavior

---

## Evaluation

### Dataset

`tst.evaluation.datasets` contains 100 stable cases:

| Category | Count | Description |
|----------|-------|-------------|
| Exact wording | 20 | Query matches stored text verbatim |
| Paraphrase | 20 | Semantic equivalent, different words |
| Pronoun follow-up | 15 | "What about that?" → context resolution |
| Conflicting preferences | 15 | Old vs new, should retrieve latest |
| Deletion handling | 10 | Deleted items must not appear |
| Scope (temp vs persistent) | 10 | STM vs LTM layer correctness |
| Dense distractors | 10 | Many similar keys, only one relevant |

### Running Evaluation

```bash
# Full evaluation with gates
python scripts/evaluate_retrieval.py

# Report only (no gate enforcement)
python scripts/evaluate_retrieval.py --report-only --output retrieval-results.json
```

### Gates (must pass)

| Metric | Threshold |
|--------|-----------|
| Recall@1 | ≥ 95% |
| Recall@3 | ≥ 98% |
| MRR | ≥ 0.95 |
| Wrong-memory rate | 0% |
| Deleted-memory leakage | 0% |
| P95 latency | ≤ 5 ms |

### Output

```json
{
  "recall_at_1": 1.0,
  "recall_at_3": 1.0,
  "mrr": 1.0,
  "wrong_memory_rate": 0.0,
  "deleted_leakage_rate": 0.0,
  "latency_p50_ms": 0.12,
  "latency_p95_ms": 0.27,
  "per_category": {
    "exact": {"recall_at_1": 1.0, "count": 20},
    "paraphrase": {"recall_at_1": 1.0, "count": 20},
    ...
  }
}
```

---

## Adding Embeddings (Future)

**Policy**: Add embeddings only if reproducible baseline shows material quality gap.

**Current baseline**: 100% Recall@1/3, 0% wrong-memory, 0% deletion leakage, 0.27 ms P95.

**Integration point**: Would replace Stage 3 (lexical) with hybrid:
```
0.6 * embedding_similarity + 0.4 * lexical_score
```
Then same rerank → dedup → budget pipeline.

**Requirements for embedding adoption**:
1. Local-only (no API calls)
2. Quantized model (< 50 MB)
3. CPU inference < 5 ms
4. Measurable quality improvement on eval dataset
5. No regression on deletion/distractor cases

---

## Configuration

### Search Parameters (per-request)

```json
{
  "layer": "ltm",
  "query": "language preference",
  "limit": 10,
  "offset": 0,
  "min_score": 0.1,
  "max_chars": 4000,
  "max_tokens": 1000
}
```

### Global Tuning (environment)

```bash
# Lexical scoring
export TST_RETRIEVAL_MIN_SCORE=0.05       # default 0.0
export TST_RETRIEVAL_MAX_CANDIDATES=2000  # default 2000

# Rerank weights (must sum to 1.0)
export TST_RETRIEVAL_W_LEXICAL=0.45
export TST_RETRIEVAL_W_PREFIX=0.20
export TST_RETRIEVAL_W_REINFORCEMENT=0.15
export TST_RETRIEVAL_W_RECENCY=0.10
export TST_RETRIEVAL_W_SOURCE=0.10

# Synonym groups (JSON file path)
export TST_RETRIEVAL_SYNONYMS=/path/to/synonyms.json
```

---

## Debugging Retrieval

```bash
# Verbose search with score breakdown
tst memory search --layer ltm --query "language" --debug

# Output includes:
# - Stage 1 exact match (if any)
# - Stage 2 candidate count per prefix
# - Stage 3 lexical scores per candidate
# - Stage 4 component scores (lexical, prefix, reinforcement, recency, source)
# - Stage 5 dedup/deletion decisions
# - Stage 6 budget selection
```