from tiny_decision_stack.backends import ScoredChoice
from tiny_decision_stack.models import DecisionRequest
from tiny_decision_stack.orchestrator import DecisionOrchestrator


class FakeSemantic:
    def __init__(self):
        self.normalize_calls = 0
        self.clarify_calls = 0

    def normalize(self, state, question, options):
        self.normalize_calls += 1
        return {
            "facts": [state],
            "constraints": [],
            "risks": [],
            "evidence_for": {},
            "evidence_against": {},
            "missing_information": [],
        }

    def clarify(self, state, normalized, question, options, probabilities):
        self.clarify_calls += 1
        return {**normalized, "constraints": ["clarified"]}


class FakeDecider:
    def __init__(self, *results):
        self.results = list(results)
        self.states = []

    def decide(self, state, question, options):
        self.states.append(state)
        return self.results.pop(0)


def req(**overrides):
    data = {
        "state": "duplicate charge",
        "question": "What should we do?",
        "options": {"refund": "refund duplicate", "escalate": "human review"},
        "confidence_threshold": 0.75,
    }
    data.update(overrides)
    return DecisionRequest(**data)


def test_direct_confident_skips_lfm():
    semantic = FakeSemantic()
    decider = FakeDecider(ScoredChoice("refund", 0.91, {"refund": 0.91, "escalate": 0.09}))
    out = DecisionOrchestrator(semantic, decider).decide(req())
    assert out.choice == "refund"
    assert out.mode == "direct"
    assert out.attempts == 1
    assert not out.abstained
    assert semantic.normalize_calls == 0


def test_low_confidence_uses_lfm_then_redecides():
    semantic = FakeSemantic()
    decider = FakeDecider(
        ScoredChoice("refund", 0.55, {"refund": 0.55, "escalate": 0.45}),
        ScoredChoice("escalate", 0.84, {"refund": 0.16, "escalate": 0.84}),
    )
    out = DecisionOrchestrator(semantic, decider).decide(req())
    assert out.choice == "escalate"
    assert out.mode == "clarified"
    assert out.attempts == 2
    assert semantic.normalize_calls == 1
    assert semantic.clarify_calls == 1


def test_still_uncertain_abstains():
    semantic = FakeSemantic()
    decider = FakeDecider(
        ScoredChoice("refund", 0.52, {"refund": 0.52, "escalate": 0.48}),
        ScoredChoice("refund", 0.58, {"refund": 0.58, "escalate": 0.42}),
    )
    out = DecisionOrchestrator(semantic, decider).decide(req(confidence_threshold=0.8))
    assert out.choice is None
    assert out.abstained
    assert out.confidence == 0.58


def test_long_state_normalizes_before_first_decision():
    semantic = FakeSemantic()
    decider = FakeDecider(ScoredChoice("refund", 0.9, {"refund": 0.9, "escalate": 0.1}))
    orchestrator = DecisionOrchestrator(semantic, decider, direct_max_chars=10)
    out = orchestrator.decide(req(state="this state is definitely longer than ten chars"))
    assert out.mode == "normalized"
    assert semantic.normalize_calls == 1
    assert isinstance(decider.states[0], dict)
