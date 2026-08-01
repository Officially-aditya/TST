"""Metric calculators shared by lightweight and model-backed evaluations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from statistics import median

from .datasets import RetrievalCase, RoutingCase


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without third-party packages."""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True, slots=True)
class RoutingObservation:
    case_id: str
    operation: str | None
    layer: str | None
    source: str
    valid: bool
    latency_ms: float
    mutated: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    case_id: str
    result_keys: tuple[str, ...]
    latency_ms: float


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def score_routing(
    cases: Iterable[RoutingCase], observations: Iterable[RoutingObservation]
) -> dict[str, object]:
    expected = {case.case_id: case for case in cases}
    observed = {observation.case_id: observation for observation in observations}
    missing = sorted(set(expected) - set(observed))
    unknown = sorted(set(observed) - set(expected))
    if missing or unknown:
        raise ValueError(f"routing observation mismatch: missing={missing}, unknown={unknown}")

    operation_correct = layer_correct = joint_correct = invalid = 0
    safe_mutations = mutation_safety_total = 0
    tier1 = tier2 = 0
    latencies: list[float] = []
    labels = sorted({case.expected_operation for case in expected.values()})
    confusion: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}
    layer_labels = sorted({case.expected_layer for case in expected.values()})
    layer_confusion: dict[str, dict[str, int]] = {label: defaultdict(int) for label in layer_labels}
    source_counts: dict[str, int] = defaultdict(int)

    for case_id, case in expected.items():
        observation = observed[case_id]
        operation_ok = observation.operation == case.expected_operation
        layer_ok = observation.layer == case.expected_layer
        operation_correct += int(operation_ok)
        layer_correct += int(layer_ok)
        joint_correct += int(operation_ok and layer_ok)
        invalid += int(not observation.valid)
        tier1 += int(observation.source in {"deterministic", "functiongemma"})
        tier2 += int(observation.source == "qwen")
        latencies.append(observation.latency_ms)
        confusion[case.expected_operation][observation.operation or "<invalid>"] += 1
        layer_confusion[case.expected_layer][observation.layer or "<invalid>"] += 1
        source_counts[observation.source] += 1
        if case.must_not_mutate:
            mutation_safety_total += 1
            safe_mutations += int(not observation.mutated)

    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        true_positive = confusion[label].get(label, 0)
        false_negative = sum(
            count for predicted, count in confusion[label].items() if predicted != label
        )
        false_positive = sum(
            row.get(label, 0)
            for expected_label, row in confusion.items()
            if expected_label != label
        )
        per_class[label] = {
            "support": true_positive + false_negative,
            "precision": _safe_ratio(true_positive, true_positive + false_positive),
            "recall": _safe_ratio(true_positive, true_positive + false_negative),
        }

    count = len(expected)
    return {
        "case_count": count,
        "operation_accuracy": _safe_ratio(operation_correct, count),
        "layer_accuracy": _safe_ratio(layer_correct, count),
        "joint_accuracy": _safe_ratio(joint_correct, count),
        "per_class": per_class,
        "operation_confusion": {
            expected_label: dict(sorted(row.items())) for expected_label, row in confusion.items()
        },
        "layer_confusion": {
            expected_label: dict(sorted(row.items()))
            for expected_label, row in layer_confusion.items()
        },
        "source_counts": dict(sorted(source_counts.items())),
        "tier1_acceptance_rate": _safe_ratio(tier1, count),
        "tier2_recovery_rate": _safe_ratio(tier2, max(1, count - tier1)),
        "invalid_output_rate": _safe_ratio(invalid, count),
        "mutation_safety_rate": _safe_ratio(safe_mutations, mutation_safety_total),
        "latency_ms": {
            "p50": median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 95),
        },
    }


def score_retrieval(
    cases: Iterable[RetrievalCase], observations: Iterable[RetrievalObservation]
) -> dict[str, object]:
    expected = {case.case_id: case for case in cases}
    observed = {observation.case_id: observation for observation in observations}
    missing = sorted(set(expected) - set(observed))
    unknown = sorted(set(observed) - set(expected))
    if missing or unknown:
        raise ValueError(f"retrieval observation mismatch: missing={missing}, unknown={unknown}")

    recall1 = recall3 = reciprocal_rank = wrong = deleted_leaks = 0.0
    answerable = wrong_denominator = deleted_count = 0
    latencies: list[float] = []
    deleted_keys_by_case = {
        case.case_id: {
            str(memory["key"]) for memory in case.memories if bool(memory.get("deleted", False))
        }
        for case in expected.values()
    }

    for case_id, case in expected.items():
        results = observed[case_id].result_keys
        latencies.append(observed[case_id].latency_ms)
        deleted_keys = deleted_keys_by_case[case_id]
        if deleted_keys:
            deleted_count += 1
            deleted_leaks += float(any(key in deleted_keys for key in results))
        if case.expected_key is None:
            wrong_denominator += 1
            wrong += float(bool(results))
            continue
        answerable += 1
        recall1 += float(bool(results) and results[0] == case.expected_key)
        recall3 += float(case.expected_key in results[:3])
        if case.expected_key in results:
            reciprocal_rank += 1.0 / (results.index(case.expected_key) + 1)
        wrong_denominator += 1
        wrong += float(bool(results) and results[0] != case.expected_key)

    return {
        "case_count": len(expected),
        "recall_at_1": _safe_ratio(recall1, answerable),
        "recall_at_3": _safe_ratio(recall3, answerable),
        "mrr": _safe_ratio(reciprocal_rank, answerable),
        "wrong_memory_rate": _safe_ratio(wrong, wrong_denominator),
        "deleted_memory_leakage_rate": _safe_ratio(deleted_leaks, deleted_count),
        "latency_ms": {
            "p50": median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 95),
        },
    }


def observations_as_dicts(
    observations: Iterable[RoutingObservation | RetrievalObservation],
) -> list[dict[str, object]]:
    return [asdict(observation) for observation in observations]
