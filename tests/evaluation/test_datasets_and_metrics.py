from __future__ import annotations

from tst.evaluation.datasets import retrieval_cases, routing_cases
from tst.evaluation.metrics import (
    RetrievalObservation,
    RoutingObservation,
    score_retrieval,
    score_routing,
)


def test_v02_dataset_sizes_are_stable() -> None:
    assert len(routing_cases) == 300
    assert len(retrieval_cases) == 100


def test_perfect_routing_score() -> None:
    observations = [
        RoutingObservation(
            case_id=case.case_id,
            operation=case.expected_operation,
            layer=case.expected_layer,
            source="deterministic",
            valid=True,
            latency_ms=0.1,
            mutated=not case.must_not_mutate,
        )
        for case in routing_cases
    ]
    metrics = score_routing(routing_cases, observations)
    assert metrics["joint_accuracy"] == 1.0
    assert metrics["mutation_safety_rate"] == 1.0
    assert metrics["invalid_output_rate"] == 0.0


def test_perfect_retrieval_score_and_no_deleted_leakage() -> None:
    observations = [
        RetrievalObservation(
            case_id=case.case_id,
            result_keys=(case.expected_key,) if case.expected_key else (),
            latency_ms=0.2,
        )
        for case in retrieval_cases
    ]
    metrics = score_retrieval(retrieval_cases, observations)
    assert metrics["recall_at_1"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["deleted_memory_leakage_rate"] == 0.0
