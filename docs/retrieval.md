# Memory retrieval

v0.2 intentionally uses no vector database. Retrieval is deterministic,
inspectable, and inexpensive enough for edge hardware.

## Pipeline

Production dispatch and evaluation use the same `RetrievalPipeline`:

1. exact canonical-key lookup;
2. bounded prefix/category candidate lookup;
3. token-based lexical scoring with conservative synonym groups;
4. reinforcement, recency, and source-confidence reranking;
5. newest-version deduplication and deletion filtering;
6. item and character-budget context selection.

The weighted score is:

```text
0.45 * lexical_similarity
+ 0.20 * key_prefix_match
+ 0.15 * reinforcement
+ 0.10 * recency
+ 0.10 * source_confidence
```

Results are filtered to the requested STM/LTM layer before ranking. A newer
deletion suppresses older live versions of the same key. Exact retrieval and
fallback search are read-only: a question is never stored as memory.

## Evaluation

`tst.evaluation.datasets` contains 100 stable cases covering exact wording,
paraphrases, pronoun-like follow-ups, conflicting and updated preferences,
deletions, temporary/persistent scope, and dense distractors.

```bash
python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --report-only --output retrieval-results.json
```

The default command exits nonzero when Recall@1/3, wrong-memory rate, deleted
memory leakage, or P95 latency misses its configured threshold. Output includes
MRR and per-category results. Add embeddings only if this reproducible baseline
shows a material quality gap.
