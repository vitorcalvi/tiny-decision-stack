from __future__ import annotations

import argparse
import json
import os
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
            cases.append(case)
    if not cases:
        raise ValueError("benchmark file contains no cases")
    return cases


def build_orchestrator() -> DecisionOrchestrator:
    device = os.getenv("MODEL_DEVICE") or None
    return DecisionOrchestrator(
        semantic=LocalLFMBackend(os.getenv("LFM_MODEL", "LiquidAI/LFM2.5-350M"), device),
        decider=LocalOpenDeciderBackend(
            os.getenv("OPENDECIDER_MODEL", "manjunathshiva/opendecider-nano"), device
        ),
        direct_max_chars=int(os.getenv("DIRECT_MAX_CHARS", "500")),
    )


def evaluate(orchestrator: DecisionOrchestrator, cases: list[dict], threshold: float) -> dict:
    answered = 0
    correct = 0
    confidences: list[float] = []
    clarified = 0

    for case in cases:
        request = DecisionRequest(
            state=case["state"],
            question=case["question"],
            options=case["options"],
            preprocess=case.get("preprocess", "auto"),
            confidence_threshold=threshold,
            clarify_on_low_confidence=case.get("clarify_on_low_confidence", True),
        )
        result = orchestrator.decide(request)
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep decision confidence thresholds on labelled JSONL cases")
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--thresholds",
        default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95",
        help="comma-separated confidence thresholds",
    )
    parser.add_argument("--target-precision", type=float, default=0.95)
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    cases = load_cases(args.dataset)
    orchestrator = build_orchestrator()
    results = [evaluate(orchestrator, cases, threshold) for threshold in thresholds]

    eligible = [
        result
        for result in results
        if result["precision"] is not None and result["precision"] >= args.target_precision
    ]
    sweet_spot = max(eligible, key=lambda item: item["coverage"], default=None)
    print(json.dumps({"results": results, "sweet_spot": sweet_spot}, indent=2))


if __name__ == "__main__":
    main()
