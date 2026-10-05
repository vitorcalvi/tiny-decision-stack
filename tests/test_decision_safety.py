"""Decision-safety regression tests: policy-owned threshold, leak-free clarification, no re-roll."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest

from tiny_decision_stack.backends import ScoredChoice
from tiny_decision_stack.models import DecisionRequest
from tiny_decision_stack.orchestrator import DecisionOrchestrator, DecisionPolicy

OPTIONS = {"refund": "refund duplicate", "escalate": "human review"}


def _normalized(state: str = "duplicate charge") -> dict:
    return {
        "facts": [state],
        "constraints": [],
        "risks": [],
        "evidence_for": {},
        "evidence_against": {},
        "missing_information": [],
    }


def _request(**overrides) -> DecisionRequest:
    data = {"state": "duplicate charge", "question": "What should we do?", "options": OPTIONS}
    data.update(overrides)
    return DecisionRequest(**data)


class _FixedConfidenceDecider:
    status = "ready"

    def __init__(self, confidence: float):
        self.confidence = confidence

    def decide(self, state, question, options):
        return ScoredChoice(
            "refund", self.confidence, {"refund": self.confidence, "escalate": 1 - self.confidence}
        )


class _HostileSemantic:
    """Model/clarifier output that tries to smuggle a permissive threshold into the pipeline."""

    status = "ready"

    def __init__(self, normalized: dict):
        self.normalized = normalized

    def normalize(self, state, question, options):
        return {**copy.deepcopy(self.normalized), "confidence_threshold": 0.0}

    def clarify(self, state, question, options, reason):
        return {
            **copy.deepcopy(self.normalized),
            "confidence_threshold": 0.0,
            "constraints": ["smuggled new evidence"],
        }


def test_threshold_owned_by_frozen_policy_and_not_mutable_by_model_output():
    policy = DecisionPolicy(confidence_threshold=0.95)
    orchestrator = DecisionOrchestrator(
        _HostileSemantic(_normalized()), _FixedConfidenceDecider(0.80), policy=policy
    )

    out = orchestrator.decide(_request(preprocess="normalize"))

    # 0.80 < 0.95: the smuggled "confidence_threshold" key never becomes the gate.
    assert out.abstained is True
    assert out.choice is None
    assert out.confidence == 0.80
    assert policy.confidence_threshold == 0.95

    with pytest.raises(FrozenInstanceError):
        policy.confidence_threshold = 0.0  # type: ignore[misc]


def test_default_policy_threshold_is_safe():
    assert DecisionPolicy().confidence_threshold == 0.75
    out = DecisionOrchestrator(_HostileSemantic(_normalized()), _FixedConfidenceDecider(0.60)).decide(
        _request()
    )
    assert out.abstained is True
    assert out.choice is None


class _RecordingSemantic:
    status = "ready"

    def __init__(self, normalized: dict):
        self.normalized = normalized
        self.clarify_calls = 0
        self.clarify_args: tuple | None = None

    def normalize(self, state, question, options):
        return copy.deepcopy(self.normalized)

    def clarify(self, state, question, options, reason):
        self.clarify_calls += 1
        self.clarify_args = (state, question, options, reason)
        return {**copy.deepcopy(self.normalized), "constraints": ["new evidence"]}


def test_clarifier_receives_only_original_input_and_reason():
    normalized = _normalized()
    semantic = _RecordingSemantic(normalized)
    orchestrator = DecisionOrchestrator(
        semantic, _FixedConfidenceDecider(0.40), policy=DecisionPolicy(confidence_threshold=0.90)
    )

    orchestrator.decide(_request())

    assert semantic.clarify_calls == 1
    assert semantic.clarify_args is not None
    state_arg, question_arg, options_arg, reason_arg = semantic.clarify_args
    assert state_arg == "duplicate charge"  # the original, unmodified input
    assert question_arg == "What should we do?"
    assert options_arg == OPTIONS
    assert isinstance(reason_arg, str) and reason_arg.strip()

    # No first-pass probabilities, scores, normalized state, or chosen label are passed.
    assert not any(isinstance(arg, dict) and arg == normalized for arg in semantic.clarify_args)
    assert all(
        not (
            isinstance(arg, dict)
            and set(arg) == set(OPTIONS)
            and all(isinstance(value, float) for value in arg.values())
        )
        for arg in semantic.clarify_args
    )
    assert "0.4" not in reason_arg
    assert "refund" not in reason_arg


class _ConfigurableSemantic:
    status = "ready"

    def __init__(self, normalized: dict, clarified: dict):
        self.normalized = normalized
        self.clarified = clarified

    def normalize(self, state, question, options):
        return copy.deepcopy(self.normalized)

    def clarify(self, state, question, options, reason):
        return copy.deepcopy(self.clarified)


class _CountingDecider:
    status = "ready"

    def __init__(self, *results: ScoredChoice):
        self.results = list(results)
        self.calls = 0

    def decide(self, state, question, options):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


_LOW = ScoredChoice("refund", 0.40, {"refund": 0.40, "escalate": 0.60})
_HIGH = ScoredChoice("escalate", 0.95, {"refund": 0.05, "escalate": 0.95})


def test_identical_clarification_abstains_without_second_pass():
    normalized = _normalized()
    semantic = _ConfigurableSemantic(normalized, copy.deepcopy(normalized))
    decider = _CountingDecider(_LOW, _HIGH)
    orchestrator = DecisionOrchestrator(
        semantic, decider, policy=DecisionPolicy(confidence_threshold=0.90)
    )

    out = orchestrator.decide(_request(preprocess="normalize"))

    assert decider.calls == 1  # never re-rolled on identical input
    assert out.attempts == 1
    assert out.abstained is True
    assert out.choice is None
    assert out.mode == "clarified"


def test_new_clarification_evidence_triggers_second_pass():
    semantic = _ConfigurableSemantic(
        _normalized(), {**_normalized(), "facts": ["ledger confirms a duplicate refund"]}
    )
    decider = _CountingDecider(_LOW, _HIGH)
    orchestrator = DecisionOrchestrator(
        semantic, decider, policy=DecisionPolicy(confidence_threshold=0.90)
    )

    out = orchestrator.decide(_request(preprocess="normalize"))

    assert decider.calls == 2
    assert out.attempts == 2
    assert out.choice == "escalate"
    assert out.mode == "clarified"