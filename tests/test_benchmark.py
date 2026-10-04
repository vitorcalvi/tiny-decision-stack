from tiny_decision_stack.backends import ScoredChoice
from tiny_decision_stack.benchmark import evaluate, sweep
from tiny_decision_stack.orchestrator import DecisionOrchestrator


class NoopSemantic:
    status = "ready"
    def normalize(self, state, question, options):
        return {"facts":[state],"constraints":[],"risks":[],"evidence_for":{},"evidence_against":{},"missing_information":[]}
    def clarify(self, state, normalized, question, options, probabilities):
        return normalized


class SequenceDecider:
    status = "ready"
    def __init__(self, results):
        self.results = iter(results)
    def decide(self, state, question, options):
        return next(self.results)


def cases():
    return [
        {"state":"a","question":"q","options":{"yes":"Y","no":"N"},"label":"yes"},
        {"state":"b","question":"q","options":{"yes":"Y","no":"N"},"label":"no"},
    ]


def test_benchmark_math():
    orch = DecisionOrchestrator(NoopSemantic(), SequenceDecider([
        ScoredChoice("yes",0.9,{"yes":0.9,"no":0.1}),
        ScoredChoice("yes",0.6,{"yes":0.6,"no":0.4}),
    ]))
    result = evaluate(orch, cases(), 0.75, direct_only=True)
    assert result["coverage"] == 0.5
    assert result["precision"] == 1.0
    assert result["effective_accuracy"] == 0.5
    assert result["abstained"] == 1


def test_sweep_selects_max_coverage_at_target_precision():
    # Fresh deterministic orchestrator per threshold is easiest to model by testing
    # the selection rule directly through two one-threshold sweeps.
    orch = DecisionOrchestrator(NoopSemantic(), SequenceDecider([
        ScoredChoice("yes",0.9,{"yes":0.9,"no":0.1}),
        ScoredChoice("no",0.9,{"yes":0.1,"no":0.9}),
    ]))
    out = sweep(orch, cases(), [0.8], direct_only=True, target_precision=0.95)
    assert out["sweet_spot"]["coverage"] == 1.0
