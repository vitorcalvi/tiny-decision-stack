from __future__ import annotations

from dataclasses import dataclass

from .backends import DecisionBackend, ScoredChoice, SemanticBackend
from .models import DecisionRequest, DecisionResponse


@dataclass
class DecisionOrchestrator:
    semantic: SemanticBackend
    decider: DecisionBackend
    default_confidence_threshold: float = 0.75
    direct_max_chars: int = 500

    def _should_normalize(self, request: DecisionRequest) -> bool:
        if request.preprocess == "normalize":
            return True
        if request.preprocess == "direct":
            return False
        # Auto mode intentionally uses a simple, explainable heuristic.
        # Short states go straight to the calibrated decider; longer states are
        # compressed into a decision-state schema first.
        return len(request.state) > self.direct_max_chars

    @staticmethod
    def _response(
        scored: ScoredChoice,
        *,
        threshold: float,
        mode: str,
        attempts: int,
        normalized_state: dict | None,
    ) -> DecisionResponse:
        abstained = scored.confidence < threshold
        return DecisionResponse(
            choice=None if abstained else scored.choice,
            confidence=scored.confidence,
            probabilities=scored.probabilities,
            abstained=abstained,
            mode=mode,
            attempts=attempts,
            normalized_state=normalized_state,
        )

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        threshold = (
            request.confidence_threshold
            if request.confidence_threshold is not None
            else self.default_confidence_threshold
        )

        normalized: dict | None = None
        if self._should_normalize(request):
            normalized = self.semantic.normalize(request.state, request.question, request.options)
            state_for_decider = normalized
            mode = "normalized"
        else:
            state_for_decider = request.state
            mode = "direct"

        first = self.decider.decide(state_for_decider, request.question, request.options)
        if first.confidence >= threshold or not request.clarify_on_low_confidence:
            return self._response(
                first,
                threshold=threshold,
                mode=mode,
                attempts=1,
                normalized_state=normalized,
            )

        # Low-confidence direct decisions get a semantic pass before retrying.
        if normalized is None:
            normalized = self.semantic.normalize(request.state, request.question, request.options)

        clarified = self.semantic.clarify(
            request.state,
            normalized,
            request.question,
            request.options,
            first.probabilities,
        )
        second = self.decider.decide(clarified, request.question, request.options)
        return self._response(
            second,
            threshold=threshold,
            mode="clarified",
            attempts=2,
            normalized_state=clarified,
        )
