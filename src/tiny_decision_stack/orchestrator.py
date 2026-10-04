from __future__ import annotations

import json
from dataclasses import dataclass

from .backends import DecisionBackend, ScoredChoice, SemanticBackend, validate_scored_choice
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

        state = request.state.strip()
        if len(state) > self.direct_max_chars:
            return True

        # Structured JSON-like inputs and concise key/value states are already good
        # decision-head inputs; messy prose with ambiguity/conflict markers gets a
        # semantic pass even when short.
        if state.startswith(("{", "[")):
            try:
                json.loads(state)
                return False
            except json.JSONDecodeError:
                pass
        if "\n" in state and all(":" in line for line in state.splitlines() if line.strip()):
            return False

        lowered = f" {state.lower()} "
        ambiguity_markers = (
            " maybe ", " unclear ", " however ", " but ", " although ",
            " conflicting ", " not sure ", " unknown ", " except ",
        )
        return any(marker in lowered for marker in ambiguity_markers)

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

        first = validate_scored_choice(
            self.decider.decide(state_for_decider, request.question, request.options), request.options
        )
        if first.confidence >= threshold or not request.clarify_on_low_confidence:
            return self._response(
                first,
                threshold=threshold,
                mode=mode,
                attempts=1,
                normalized_state=normalized,
            )

        if normalized is None:
            normalized = self.semantic.normalize(request.state, request.question, request.options)

        clarified = self.semantic.clarify(
            request.state,
            normalized,
            request.question,
            request.options,
            first.probabilities,
        )
        second = validate_scored_choice(
            self.decider.decide(clarified, request.question, request.options), request.options
        )
        return self._response(
            second,
            threshold=threshold,
            mode="clarified",
            attempts=2,
            normalized_state=clarified,
        )
