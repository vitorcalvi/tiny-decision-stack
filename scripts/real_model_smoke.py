from __future__ import annotations

from tiny_decision_stack.backends import LocalLFMBackend, LocalOpenDeciderBackend
from tiny_decision_stack.models import DecisionRequest
from tiny_decision_stack.orchestrator import DecisionOrchestrator


def main() -> None:
    semantic = LocalLFMBackend(device="cpu")
    decider = LocalOpenDeciderBackend(device="cpu")
    assert semantic.status == "not_loaded"
    assert decider.status == "not_loaded"

    stack = DecisionOrchestrator(
        semantic=semantic,
        decider=decider,
        direct_max_chars=1,
    )
    cases = [
        (
            "A customer reports a duplicate card charge. Logs confirm the same order was charged twice.",
            {
                "refund": "Refund a verified duplicate payment",
                "escalate": "Send uncertain cases to human review",
            },
        ),
        (
            "The payment is duplicated but authorization metadata is unclear.",
            {
                "refund": "Refund only a verified duplicate",
                "review": "Require human review before action",
            },
        ),
    ]

    for state, options in cases:
        result = stack.decide(
            DecisionRequest(
                state=state,
                question="What action should the workflow take?",
                options=options,
                preprocess="normalize",
                confidence_threshold=0.0,
                clarify_on_low_confidence=False,
            )
        )
        assert result.choice in options
        assert result.mode == "normalized"
        assert result.normalized_state is not None
        assert set(result.probabilities) == set(options)
        assert all(0.0 <= value <= 1.0 for value in result.probabilities.values())
        assert abs(sum(result.probabilities.values()) - 1.0) <= 0.02
        print(result.model_dump_json())

    assert semantic.status == "ready"
    assert decider.status == "ready"
    print("REAL_MODEL_SMOKE_OK")


if __name__ == "__main__":
    main()
