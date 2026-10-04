from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from statistics import mean

from .backends import LocalLFMBackend, LocalOpenDeciderBackend
from .models import DecisionRequest
from .orchestrator import DecisionOrchestrator


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            required = {"state", "question", "options", "label"}
            missing = required - case.keys()
            if missing:
                raise ValueError(f"line {line_no}: missing {sorted(missing)}")
            if case["label"] not in case["options"]:
                raise ValueError(f"line {line_no}: label must be one of options")
            cases.append(case)
    if not cases:
        raise ValueError("benchmark file contains no cases")
    return cases


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_orchestrator() -> DecisionOrchestrator:
    device = os.getenv("MODEL_DEVICE") or None
    return DecisionOrchestrator(
        semantic=LocalLFMBackend(os.getenv("LFM_MODEL", "LiquidAI/LFM2.5-350M"), device),
        decider=LocalOpenDeciderBackend(
            os.getenv("OPENDECIDER_MODEL", "manjunathshiva/opendecider-nano"), device
        ),
        direct_max_chars=int(os.getenv("DIRECT_MAX_CHARS", "500")),
    )


def evaluate(
    orchestrator: DecisionOrchestrator,
    cases: list[dict],
    threshold: float,
    *,
    direct_only: bool = False,
) -> dict:
    answered = 0
    correct = 0
    confidences: list[float] = []
    clarified = 0
    latencies_ms: list[float] = []

    for case in cases:
        request = DecisionRequest(
            state=case["state"],
            question=case["question"],
            options=case["options"],
            preprocess="direct" if direct_only else case.get("preprocess", "auto"),
            confidence_threshold=threshold,
            clarify_on_low_confidence=False if direct_only else case.get("clarify_on_low_confidence", True),
        )
        started = time.perf_counter()
        result = orchestrator.decide(request)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        confidences.append(result.confidence)
        clarified += int(result.mode == "clarified")
        if not result.abstained:
            answered += 1
            correct += int(result.choice == case["label"])

    total = len(cases)
    return {
        "threshold": threshold,
        "cases": total,
        "answered": answered,
        "abstained": total - answered,
        "coverage": answered / total,
        "precision": (correct / answered) if answered else None,
        "effective_accuracy": correct / total,
        "mean_final_confidence": mean(confidences),
        "clarification_rate": clarified / total,
        "mean_latency_ms": mean(latencies_ms),
    }


def sweep(orchestrator, cases, thresholds, *, direct_only=False, target_precision=0.95):
    results = [evaluate(orchestrator, cases, threshold, direct_only=direct_only) for threshold in thresholds]
    eligible = [r for r in results if r["precision"] is not None and r["precision"] >= target_precision]
    sweet_spot = max(eligible, key=lambda item: item["coverage"], default=None)
    return {"results": results, "sweet_spot": sweet_spot}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds on labelled JSONL decisions")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument(
        "--compare-direct",
        action="store_true",
        help="also evaluate OpenDecider-only behavior by forcing direct mode and disabling clarification",
    )
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    if any(not 0 <= x <= 1 for x in thresholds):
        raise ValueError("thresholds must be within [0, 1]")
    cases = load_cases(args.dataset)
    orchestrator = build_orchestrator()
    payload = {
        "dataset_sha256": dataset_sha256(args.dataset),
        "target_precision": args.target_precision,
        "stack": sweep(orchestrator, cases, thresholds, target_precision=args.target_precision),
    }
    if args.compare_direct:
        payload["direct_only"] = sweep(
            orchestrator, cases, thresholds, direct_only=True, target_precision=args.target_precision
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
