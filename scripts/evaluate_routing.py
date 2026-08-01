#!/usr/bin/env python3
"""Run the model-free routing safety baseline over all 300 labelled cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tst.evaluation.datasets import routing_cases
from tst.evaluation.metrics import RoutingObservation, score_routing
from tst.routing.router import ActionRouter


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic routing and safe fallback. Model-backed tiers "
            "can use the same dataset/metric API from tst.evaluation."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("routing-results.json"))
    parser.add_argument("--min-operation-accuracy", type=float, default=0.60)
    parser.add_argument("--min-layer-accuracy", type=float, default=0.60)
    parser.add_argument("--min-joint-accuracy", type=float, default=0.60)
    parser.add_argument("--min-mutation-safety", type=float, default=1.0)
    parser.add_argument("--max-invalid-output-rate", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float, default=5.0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    router = ActionRouter()
    observations: list[RoutingObservation] = []
    decisions: list[dict[str, object]] = []
    for case in routing_cases:
        started = time.perf_counter()
        decision = router.route(case.text)
        latency_ms = (time.perf_counter() - started) * 1000.0
        observations.append(
            RoutingObservation(
                case_id=case.case_id,
                operation=decision.operation,
                layer=decision.layer,
                source=decision.source,
                valid=True,
                latency_ms=latency_ms,
                mutated=decision.mutates_memory,
            )
        )
        decisions.append(
            {
                "case_id": case.case_id,
                "expected": {
                    "operation": case.expected_operation,
                    "layer": case.expected_layer,
                },
                "decision": decision.model_dump(mode="json"),
                "valid": True,
                "source": decision.source,
                "mutated": decision.mutates_memory,
                "latency_ms": latency_ms,
            }
        )

    metrics = score_routing(routing_cases, observations)
    gates = {
        "operation_accuracy": metrics["operation_accuracy"] >= args.min_operation_accuracy,
        "layer_accuracy": metrics["layer_accuracy"] >= args.min_layer_accuracy,
        "joint_accuracy": metrics["joint_accuracy"] >= args.min_joint_accuracy,
        "mutation_safety_rate": (metrics["mutation_safety_rate"] >= args.min_mutation_safety),
        "invalid_output_rate": (metrics["invalid_output_rate"] <= args.max_invalid_output_rate),
        "latency_p95_ms": metrics["latency_ms"]["p95"] <= args.max_p95_ms,
    }
    document = {
        "schema_version": 1,
        "mode": "deterministic_and_safe_fallback",
        "metrics": metrics,
        "gates": {
            "passed": all(gates.values()),
            "checks": gates,
            "report_only": args.report_only,
        },
        "decisions": decisions,
    }
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document["metrics"], indent=2))
    print(f"Saved {len(decisions)} cases to {args.output}")
    return 0 if args.report_only or all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
