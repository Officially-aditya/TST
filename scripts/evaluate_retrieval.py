#!/usr/bin/env python3
"""Evaluate the v0.2 canonical/prefix/lexical retrieval pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from tst.evaluation.datasets import retrieval_cases
from tst.evaluation.metrics import RetrievalObservation, score_retrieval
from tst.memory.pipeline import RetrievalPipeline
from tst.memory.retrieval import LexicalMemoryRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("retrieval-results.json"))
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument("--min-recall-at-1", type=float, default=0.85)
    parser.add_argument("--min-recall-at-3", type=float, default=0.95)
    parser.add_argument("--max-wrong-memory-rate", type=float, default=0.10)
    parser.add_argument("--max-deleted-leakage-rate", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float, default=20.0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    pipeline = RetrievalPipeline(retriever=LexicalMemoryRetriever(min_score=args.min_score))
    observations: list[RetrievalObservation] = []
    details: list[dict[str, object]] = []
    by_category: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for case in retrieval_cases:
        # Exact cases exercise step one; all other categories must earn their
        # result from category/key tokens, lexical similarity, and reranking.
        canonical_key = case.expected_key if case.category == "exact" else None
        now_ms = max((int(memory["updated_at"]) for memory in case.memories), default=0) + 1_000
        started = time.perf_counter()
        prefix = "session:" if case.layer == "stm" else "user:default:preference:"
        retrieval = pipeline.retrieve(
            case.query,
            case.memories,
            canonical_key=canonical_key,
            prefix=prefix,
            layer=case.layer,
            top_k=3,
            max_characters=2_000,
            now_ms=now_ms,
        )
        ranked = retrieval.ranked
        latency_ms = (time.perf_counter() - started) * 1_000.0
        keys = tuple(item.record.key for item in ranked)
        observations.append(RetrievalObservation(case.case_id, keys, latency_ms))
        correct = (
            (not keys) if case.expected_key is None else bool(keys and keys[0] == case.expected_key)
        )
        by_category[case.category].append((correct, latency_ms))
        details.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "query": case.query,
                "expected_key": case.expected_key,
                "results": [
                    {
                        "key": item.record.key,
                        "score": item.score,
                        "lexical_similarity": item.lexical_similarity,
                        "key_prefix_match": item.key_prefix_match,
                        "recency": item.recency,
                    }
                    for item in ranked
                ],
                "latency_ms": latency_ms,
            }
        )

    metrics = score_retrieval(retrieval_cases, observations)
    metrics["by_category"] = {
        category: {
            "count": len(values),
            "top1_accuracy": sum(int(correct) for correct, _ in values) / len(values),
            "mean_latency_ms": sum(latency for _, latency in values) / len(values),
        }
        for category, values in sorted(by_category.items())
    }
    document = {
        "schema_version": 1,
        "min_score": args.min_score,
        "metrics": metrics,
        "cases": details,
    }
    gates = {
        "recall_at_1": metrics["recall_at_1"] >= args.min_recall_at_1,
        "recall_at_3": metrics["recall_at_3"] >= args.min_recall_at_3,
        "wrong_memory_rate": metrics["wrong_memory_rate"] <= args.max_wrong_memory_rate,
        "deleted_memory_leakage_rate": (
            metrics["deleted_memory_leakage_rate"] <= args.max_deleted_leakage_rate
        ),
        "latency_p95_ms": metrics["latency_ms"]["p95"] <= args.max_p95_ms,
    }
    document["gates"] = {
        "passed": all(gates.values()),
        "checks": gates,
        "report_only": args.report_only,
    }
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {len(details)} cases to {args.output}")
    return 0 if args.report_only or all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
