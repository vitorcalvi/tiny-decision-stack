from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .backends import BackendValidationError, ScoredChoice, validate_scored_choice
from .models import DecisionRequest
from .orchestrator import DecisionOrchestrator

DEFAULT_THRESHOLDS = (0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
BOOTSTRAP_SEED = 20261004


@dataclass
class StageScore:
    choice: str
    confidence: float
    probabilities: dict[str, float]

    @classmethod
    def from_scored(cls, scored: ScoredChoice) -> "StageScore":
        return cls(scored.choice, scored.confidence, dict(scored.probabilities))


@dataclass
class CaseTrace:
    id: str
    task: str
    label: str
    variant: str
    initial_mode: str
    initial_normalized: bool
    first: StageScore | None
    first_latency_ms: float
    first_error: str | None = None
    clarify_enabled: bool = True
    second: StageScore | None = None
    second_latency_ms: float = 0.0
    second_error: str | None = None
    no_new_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulatedOutcome:
    id: str
    task: str
    label: str
    variant: str
    threshold: float
    choice: str | None
    raw_choice: str | None
    confidence: float | None
    answered: bool
    correct: bool
    abstained: bool
    model_error: bool
    error_type: str | None
    used_clarification: bool
    initial_normalized: bool
    mode: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "task", "state", "question", "options", "label"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
            if row["label"] not in row["options"]:
                raise ValueError(f"{path}:{line_no}: label is not an option")
            if len(row["options"]) < 2:
                raise ValueError(f"{path}:{line_no}: fewer than two options")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no benchmark cases")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate stable IDs")
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_suite(benchmark_dir: Path) -> dict[str, Any]:
    manifest_path = benchmark_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration_path = benchmark_dir / "calibration.jsonl"
    holdout_path = benchmark_dir / "holdout.jsonl"
    calibration = load_cases(calibration_path)
    holdout = load_cases(holdout_path)

    expected = manifest["files"]
    for path in (calibration_path, holdout_path):
        spec = expected[path.name]
        actual_sha = sha256_file(path)
        if actual_sha != spec["sha256"]:
            raise ValueError(f"{path.name} SHA256 drift: {actual_sha} != {spec['sha256']}")
        actual_count = len(calibration) if path == calibration_path else len(holdout)
        if actual_count != spec["count"]:
            raise ValueError(f"{path.name} count drift: {actual_count} != {spec['count']}")

    cal_ids = {row["id"] for row in calibration}
    hold_ids = {row["id"] for row in holdout}
    overlap = cal_ids & hold_ids
    if overlap:
        raise ValueError(f"calibration/holdout ID leakage: {sorted(overlap)[:5]}")

    # Catch copied examples even if IDs differ.
    def fingerprint(row: dict[str, Any]) -> str:
        payload = json.dumps(
            [row["task"], row["state"], row["question"], row["options"], row["label"]],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    cal_fp = {fingerprint(row) for row in calibration}
    hold_fp = {fingerprint(row) for row in holdout}
    duplicate_content = cal_fp & hold_fp
    if duplicate_content:
        raise ValueError("calibration/holdout content leakage detected")

    holdout_by_task = manifest["protocol"]["holdout_by_task"]
    actual_holdout_by_task: dict[str, int] = {}
    for row in holdout:
        actual_holdout_by_task[row["task"]] = actual_holdout_by_task.get(row["task"], 0) + 1
    if actual_holdout_by_task != holdout_by_task:
        raise ValueError(f"holdout task counts drift: {actual_holdout_by_task} != {holdout_by_task}")
    return manifest


def _error_name(exc: BaseException) -> str:
    return type(exc).__name__


def _attempt_decide(
    orchestrator: DecisionOrchestrator,
    state: str | dict[str, Any],
    request: DecisionRequest,
) -> tuple[ScoredChoice | None, str | None, float]:
    """Run a decision stage; model-contract failures are benchmark outcomes, infrastructure failures still raise."""
    started = time.perf_counter()
    try:
        scored = validate_scored_choice(
            orchestrator.decider.decide(state, request.question, request.options), request.options
        )
        return scored, None, (time.perf_counter() - started) * 1000.0
    except BackendValidationError as exc:
        return None, _error_name(exc), (time.perf_counter() - started) * 1000.0


def trace_case_pair(
    orchestrator: DecisionOrchestrator,
    case: dict[str, Any],
    *,
    max_threshold: float,
) -> tuple[CaseTrace, CaseTrace]:
    """Trace direct-only and full-stack once; contract-invalid model output becomes an explicit case error."""
    request = DecisionRequest(
        state=case["state"],
        question=case["question"],
        options=case["options"],
        preprocess=case.get("preprocess", "auto"),
        clarify_on_low_confidence=case.get("clarify_on_low_confidence", True),
        confidence_threshold=0.0,
    )

    direct_request = request.model_copy(update={"preprocess": "direct", "clarify_on_low_confidence": False})
    direct_first, direct_error, direct_latency = _attempt_decide(
        orchestrator, direct_request.state, direct_request
    )
    direct_trace = CaseTrace(
        id=case["id"], task=case["task"], label=case["label"], variant="direct_only",
        initial_mode="direct", initial_normalized=False,
        first=StageScore.from_scored(direct_first) if direct_first else None,
        first_latency_ms=direct_latency, first_error=direct_error, clarify_enabled=False,
    )

    should_normalize = orchestrator._should_normalize(request)
    normalized: dict[str, Any] | None = None
    if should_normalize:
        started = time.perf_counter()
        try:
            normalized = orchestrator.semantic.normalize(request.state, request.question, request.options)
            full_first, full_error, _ = _attempt_decide(orchestrator, normalized, request)
        except BackendValidationError as exc:
            full_first, full_error = None, _error_name(exc)
        full_first_latency = (time.perf_counter() - started) * 1000.0
        initial_mode = "normalized"
    else:
        full_first, full_error, full_first_latency = direct_first, direct_error, direct_latency
        initial_mode = "direct"

    full_trace = CaseTrace(
        id=case["id"], task=case["task"], label=case["label"], variant="full_stack",
        initial_mode=initial_mode, initial_normalized=should_normalize,
        first=StageScore.from_scored(full_first) if full_first else None,
        first_latency_ms=full_first_latency, first_error=full_error,
        clarify_enabled=request.clarify_on_low_confidence,
    )

    # Precompute the retry once if any candidate threshold could invoke it.
    if full_first is not None and request.clarify_on_low_confidence and full_first.confidence < max_threshold:
        started = time.perf_counter()
        try:
            if normalized is None:
                normalized = orchestrator.semantic.normalize(request.state, request.question, request.options)
            clarified = orchestrator.clarify_state(request, normalized)
            if orchestrator._has_new_evidence(clarified, normalized):
                full_second, second_error, _ = _attempt_decide(orchestrator, clarified, request)
                full_trace.second = StageScore.from_scored(full_second) if full_second else None
                full_trace.second_error = second_error
            else:
                full_trace.no_new_evidence = True
        except BackendValidationError as exc:
            full_trace.second_error = _error_name(exc)
        full_trace.second_latency_ms = (time.perf_counter() - started) * 1000.0

    return direct_trace, full_trace


def _error_outcome(
    trace: CaseTrace, threshold: float, error_type: str, *, used_clarification: bool, latency_ms: float
) -> SimulatedOutcome:
    return SimulatedOutcome(
        id=trace.id, task=trace.task, label=trace.label, variant=trace.variant, threshold=threshold,
        choice=None, raw_choice=None, confidence=None, answered=False, correct=False, abstained=False,
        model_error=True, error_type=error_type, used_clarification=used_clarification,
        initial_normalized=trace.initial_normalized,
        mode="clarified" if used_clarification else trace.initial_mode, latency_ms=latency_ms,
    )


def simulate(trace: CaseTrace, threshold: float) -> SimulatedOutcome:
    if trace.first is None:
        return _error_outcome(
            trace, threshold, trace.first_error or "UnknownFirstStageError",
            used_clarification=False, latency_ms=trace.first_latency_ms,
        )

    use_second = (
        trace.variant == "full_stack"
        and trace.clarify_enabled
        and trace.first.confidence < threshold
    )
    if use_second:
        if trace.no_new_evidence:
            # Clarifier restated the first-pass input: abstain instead of re-rolling.
            return SimulatedOutcome(
                id=trace.id, task=trace.task, label=trace.label, variant=trace.variant, threshold=threshold,
                choice=None, raw_choice=None, confidence=trace.first.confidence, answered=False,
                correct=False, abstained=True, model_error=False, error_type=None,
                used_clarification=True, initial_normalized=trace.initial_normalized,
                mode="clarified", latency_ms=trace.first_latency_ms + trace.second_latency_ms,
            )
        if trace.second_error is not None:
            return _error_outcome(
                trace, threshold, trace.second_error, used_clarification=True,
                latency_ms=trace.first_latency_ms + trace.second_latency_ms,
            )
        if trace.second is None:
            return _error_outcome(
                trace, threshold, "MissingPrecomputedClarification", used_clarification=True,
                latency_ms=trace.first_latency_ms + trace.second_latency_ms,
            )
        stage = trace.second
    else:
        stage = trace.first

    abstained = stage.confidence < threshold
    choice = None if abstained else stage.choice
    return SimulatedOutcome(
        id=trace.id, task=trace.task, label=trace.label, variant=trace.variant, threshold=threshold,
        choice=choice, raw_choice=stage.choice, confidence=stage.confidence, answered=not abstained,
        correct=(not abstained and stage.choice == trace.label), abstained=abstained, model_error=False,
        error_type=None, used_clarification=use_second, initial_normalized=trace.initial_normalized,
        mode="clarified" if use_second else trace.initial_mode,
        latency_ms=trace.first_latency_ms + (trace.second_latency_ms if use_second else 0.0),
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize_outcomes(outcomes: list[SimulatedOutcome]) -> dict[str, Any]:
    n = len(outcomes)
    answered = sum(item.answered for item in outcomes)
    correct = sum(item.correct for item in outcomes)
    abstained = sum(item.abstained for item in outcomes)
    model_errors = sum(item.model_error for item in outcomes)
    effective_accuracy = correct / n if n else 0.0
    precision = correct / answered if answered else None
    latencies = [item.latency_ms for item in outcomes]
    confidences = [item.confidence for item in outcomes if item.confidence is not None]
    error_types: dict[str, int] = {}
    for item in outcomes:
        if item.error_type:
            error_types[item.error_type] = error_types.get(item.error_type, 0) + 1
    result: dict[str, Any] = {
        "n": n,
        "correct": correct,
        "answered": answered,
        "abstained": abstained,
        "model_errors": model_errors,
        "model_error_rate": model_errors / n if n else 0.0,
        "model_error_types": dict(sorted(error_types.items())),
        "accuracy": effective_accuracy,
        "effective_accuracy": effective_accuracy,
        "effective_accuracy_wilson95": wilson_interval(correct, n),
        "selective_precision": precision,
        "selective_precision_wilson95": wilson_interval(correct, answered) if answered else None,
        "coverage": answered / n if n else 0.0,
        "mean_confidence": statistics.fmean(confidences) if confidences else None,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "initial_normalization_rate": sum(item.initial_normalized for item in outcomes) / n if n else 0.0,
        "clarification_rate": sum(item.used_clarification for item in outcomes) / n if n else 0.0,
    }
    return result


def summarize_by_task(outcomes: list[SimulatedOutcome]) -> dict[str, dict[str, Any]]:
    tasks = sorted({item.task for item in outcomes})
    return {task: summarize_outcomes([item for item in outcomes if item.task == task]) for task in tasks}


def select_threshold(
    traces: list[CaseTrace],
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    target_precision: float = 0.90,
    role: str = "calibration",
) -> dict[str, Any]:
    if role != "calibration":
        raise ValueError("threshold selection is permitted only on the calibration split")
    candidates = sorted(set(float(value) for value in thresholds))
    if not candidates or candidates[0] < 0.0 or candidates[-1] > 1.0:
        raise ValueError("threshold candidates must be within [0, 1]")
    sweep: list[dict[str, Any]] = []
    for threshold in candidates:
        outcomes = [simulate(trace, threshold) for trace in traces]
        metrics = summarize_outcomes(outcomes)
        sweep.append({"threshold": threshold, **metrics})

    eligible = [row for row in sweep if row["selective_precision"] is not None and row["selective_precision"] >= target_precision]
    if eligible:
        chosen = max(eligible, key=lambda row: (row["coverage"], -row["threshold"]))
        target_met = True
        reason = "maximum calibration coverage meeting target selective precision"
    else:
        # Deterministic fallback: strongest precision, then coverage, then lower threshold.
        chosen = max(
            sweep,
            key=lambda row: (
                -1.0 if row["selective_precision"] is None else row["selective_precision"],
                row["coverage"],
                -row["threshold"],
            ),
        )
        target_met = False
        reason = "target precision unreachable on calibration; selected highest observed precision"
    return {
        "target_precision": target_precision,
        "target_met": target_met,
        "chosen_threshold": chosen["threshold"],
        "chosen_calibration_metrics": chosen,
        "selection_reason": reason,
        "sweep": sweep,
    }


def paired_bootstrap_difference(
    full: list[SimulatedOutcome],
    direct: list[SimulatedOutcome],
    *,
    reps: int = 5000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    full_by_id = {item.id: item for item in full}
    direct_by_id = {item.id: item for item in direct}
    if set(full_by_id) != set(direct_by_id):
        raise ValueError("paired bootstrap requires identical stable IDs")
    ids = sorted(full_by_id)
    pairs = [(1.0 if full_by_id[i].correct else 0.0, 1.0 if direct_by_id[i].correct else 0.0) for i in ids]
    observed = statistics.fmean(a - b for a, b in pairs) if pairs else 0.0
    rng = random.Random(seed)
    diffs: list[float] = []
    n = len(pairs)
    for _ in range(reps):
        total = 0.0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            total += a - b
        diffs.append(total / n if n else 0.0)
    return {
        "metric": "effective_accuracy(full_stack) - effective_accuracy(direct_only)",
        "observed_difference": observed,
        "bootstrap_reps": reps,
        "seed": seed,
        "ci95": [_percentile(diffs, 0.025), _percentile(diffs, 0.975)],
    }


def assert_trace_simulation_matches_orchestrator(
    orchestrator: DecisionOrchestrator,
    case: dict[str, Any],
    full_trace: CaseTrace,
    thresholds: Iterable[float],
) -> None:
    """Test/helper: verify cached trace simulation matches real orchestrator semantics."""
    for threshold in thresholds:
        request = DecisionRequest(
            state=case["state"],
            question=case["question"],
            options=case["options"],
            preprocess=case.get("preprocess", "auto"),
            clarify_on_low_confidence=case.get("clarify_on_low_confidence", True),
            confidence_threshold=float(threshold),
        )
        real = orchestrator.decide(request)
        simulated = simulate(full_trace, float(threshold))
        assert real.choice == simulated.choice
        assert real.abstained == simulated.abstained
        assert real.mode == simulated.mode
        assert math.isclose(real.confidence, simulated.confidence, rel_tol=0.0, abs_tol=1e-12)
