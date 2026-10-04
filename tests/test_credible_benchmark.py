from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tiny_decision_stack.backends import ScoredChoice, SemanticOutputError
from tiny_decision_stack.credible_benchmark import (
    CaseTrace,
    SimulatedOutcome,
    StageScore,
    assert_trace_simulation_matches_orchestrator,
    load_cases,
    paired_bootstrap_difference,
    select_threshold,
    simulate,
    trace_case_pair,
    verify_suite,
    wilson_interval,
)
from tiny_decision_stack.orchestrator import DecisionOrchestrator

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "credible-v1"


class DeterministicSemantic:
    status = "ready"

    def normalize(self, state, question, options):
        return {
            "facts": [state],
            "constraints": [],
            "risks": [],
            "evidence_for": {},
            "evidence_against": {},
            "missing_information": [],
        }

    def clarify(self, state, normalized, question, options, probabilities):
        return {**normalized, "constraints": ["clarified"]}


class DeterministicDecider:
    status = "ready"

    def decide(self, state, question, options):
        keys = list(options)
        a, b = keys[0], keys[1]
        if isinstance(state, dict) and state.get("constraints") == ["clarified"]:
            return ScoredChoice(a, 0.90, {a: 0.90, b: 0.10})
        if isinstance(state, dict):
            return ScoredChoice(b, 0.65, {a: 0.35, b: 0.65})
        return ScoredChoice(b, 0.60, {a: 0.40, b: 0.60})


def _case(*, preprocess="auto", state="plain state"):
    return {
        "id": f"case:{preprocess}:{len(state)}",
        "task": "unit",
        "state": state,
        "question": "Choose",
        "options": {"a": "A", "b": "B"},
        "label": "a",
        "preprocess": preprocess,
        "clarify_on_low_confidence": True,
    }


def test_frozen_suite_integrity_and_counts():
    manifest = verify_suite(BENCH)
    calibration = load_cases(BENCH / "calibration.jsonl")
    holdout = load_cases(BENCH / "holdout.jsonl")
    assert len(calibration) == 354
    assert len(holdout) == 600
    assert manifest["protocol"]["holdout_by_task"] == {
        "arc_challenge": 200,
        "banking77": 200,
        "boolq": 200,
    }
    assert not ({row["id"] for row in calibration} & {row["id"] for row in holdout})

    bank = [row for row in holdout if row["task"] == "banking77"]
    assert len({row["label"] for row in bank}) == 77
    assert all(len(row["options"]) == 77 for row in bank)
    assert all(len(row["options"]) >= 2 and row["label"] in row["options"] for row in holdout)


def test_hash_drift_is_rejected(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    for name in ("manifest.json", "calibration.jsonl", "holdout.jsonl"):
        (suite / name).write_bytes((BENCH / name).read_bytes())
    with (suite / "holdout.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="SHA256 drift"):
        verify_suite(suite)


def test_dataset_builder_selection_is_deterministic():
    path = ROOT / "scripts" / "build_credible_dataset.py"
    spec = importlib.util.spec_from_file_location("credible_data_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"id": f"x:{i}"} for i in range(50)]
    first = [r["id"] for r in module.select_hash(rows, 10, salt="test")]
    second = [r["id"] for r in module.select_hash(list(reversed(rows)), 10, salt="test")]
    assert first == second
    assert first == [
        "x:39", "x:1", "x:46", "x:37", "x:29", "x:45", "x:12", "x:5", "x:24", "x:19"
    ]


def test_wilson_interval_known_value():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038315, abs=1e-6)
    assert high == pytest.approx(0.5961685, abs=1e-6)
    assert wilson_interval(0, 0) is None


def test_threshold_selection_refuses_holdout_and_uses_calibration_only():
    traces = [
        CaseTrace(
            id=f"x{i}", task="unit", label="a", variant="direct_only", initial_mode="direct",
            initial_normalized=False,
            first=StageScore("a" if i < 8 else "b", 0.90 if i < 8 else 0.55, {"a": 0.9, "b": 0.1}),
            first_latency_ms=1.0,
        )
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="calibration"):
        select_threshold(traces, thresholds=(0.0, 0.6), target_precision=0.9, role="holdout")
    selected = select_threshold(traces, thresholds=(0.0, 0.6), target_precision=0.9, role="calibration")
    assert selected["chosen_threshold"] == 0.6
    assert selected["target_met"] is True


def test_cached_trace_matches_orchestrator_direct_and_clarified_paths():
    case = _case(preprocess="direct")
    orchestrator = DecisionOrchestrator(DeterministicSemantic(), DeterministicDecider())
    _, full = trace_case_pair(orchestrator, case, max_threshold=0.99)
    assert full.initial_mode == "direct"
    assert full.second is not None
    fresh = DecisionOrchestrator(DeterministicSemantic(), DeterministicDecider())
    assert_trace_simulation_matches_orchestrator(fresh, case, full, (0.50, 0.70, 0.95))
    assert simulate(full, 0.50).mode == "direct"
    assert simulate(full, 0.70).mode == "clarified"
    assert simulate(full, 0.95).abstained is True


def test_cached_trace_matches_orchestrator_normalized_path():
    case = _case(preprocess="normalize", state="ambiguous state")
    orchestrator = DecisionOrchestrator(DeterministicSemantic(), DeterministicDecider())
    _, full = trace_case_pair(orchestrator, case, max_threshold=0.99)
    assert full.initial_mode == "normalized"
    fresh = DecisionOrchestrator(DeterministicSemantic(), DeterministicDecider())
    assert_trace_simulation_matches_orchestrator(fresh, case, full, (0.50, 0.70, 0.95))



class BadClarificationSemantic(DeterministicSemantic):
    def clarify(self, state, normalized, question, options, probabilities):
        raise SemanticOutputError("invalid semantic schema")


def test_model_contract_failure_is_counted_not_crash():
    case = _case(preprocess="direct")
    orchestrator = DecisionOrchestrator(BadClarificationSemantic(), DeterministicDecider())
    _, full = trace_case_pair(orchestrator, case, max_threshold=0.99)
    outcome = simulate(full, 0.70)
    assert outcome.model_error is True
    assert outcome.error_type == "SemanticOutputError"
    assert outcome.answered is False
    assert outcome.abstained is False

def _outcome(case_id: str, variant: str, correct: bool) -> SimulatedOutcome:
    return SimulatedOutcome(
        id=case_id, task="unit", label="a", variant=variant, threshold=0.5,
        choice="a" if correct else "b", raw_choice="a" if correct else "b", confidence=0.9,
        answered=True, correct=correct, abstained=False, model_error=False, error_type=None,
        used_clarification=False, initial_normalized=False, mode="direct", latency_ms=1.0,
    )


def test_paired_bootstrap_is_deterministic_and_paired():
    full = [_outcome(str(i), "full_stack", i < 8) for i in range(10)]
    direct = [_outcome(str(i), "direct_only", i < 6) for i in range(10)]
    a = paired_bootstrap_difference(full, direct, reps=1000, seed=123)
    b = paired_bootstrap_difference(full, direct, reps=1000, seed=123)
    assert a == b
    assert a["observed_difference"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="identical stable IDs"):
        paired_bootstrap_difference(full, direct[:-1], reps=10, seed=1)
