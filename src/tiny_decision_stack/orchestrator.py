from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any

from .backends import DecisionBackend, ScoredChoice, SemanticBackend, validate_scored_choice
from .models import DecisionRequest, DecisionResponse

SAFE_DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_DIRECT_MAX_CHARS = 500
CLARIFY_REASON = (
    "The first decision pass was inconclusive. Re-express only the decision-relevant "
    "evidence that can be derived from the original input; do not choose an option."
)


@dataclass(frozen=True)
class DecisionPolicy:
    """Caller-owned policy. Frozen so model or clarifier output can never mutate it."""

    confidence_threshold: float = SAFE_DEFAULT_CONFIDENCE_THRESHOLD
    direct_max_chars: int = DEFAULT_DIRECT_MAX_CHARS

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if self.direct_max_chars < 1:
            raise ValueError("direct_max_chars must be >= 1")


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


class DecisionOrchestrator:
    def __init__(
        self,
        semantic: SemanticBackend,
        decider: DecisionBackend,
        policy: DecisionPolicy | None = None,
        *,
        default_confidence_threshold: float | None = None,
        direct_max_chars: int | None = None,
    ) -> None:
        self.semantic = semantic
        self.decider = decider
        if policy is None:
            policy = DecisionPolicy(
                confidence_threshold=(
                    SAFE_DEFAULT_CONFIDENCE_THRESHOLD
                    if default_confidence_threshold is None
                    else default_confidence_threshold
                ),
                direct_max_chars=(
                    DEFAULT_DIRECT_MAX_CHARS if direct_max_chars is None else direct_max_chars
                ),
            )
        self.policy = policy

    def _should_normalize(self, request: DecisionRequest) -> bool:
        if request.preprocess == "normalize":
            return True
        if request.preprocess == "direct":
            return False

        state = request.state.strip()
        if len(state) > self.policy.direct_max_chars:
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

    def clarify_state(self, request: DecisionRequest, normalized: dict[str, Any]) -> dict[str, Any]:
        """Ask the semantic backend for new evidence, withholding scores and labels.

        New-contract backends receive only the original input and the reason. Legacy
        five-positional backends receive the reason in the slot that used to hold the
        first-pass probabilities, so no score, probability, or chosen label leaks.
        """
        clarify = self.semantic.clarify
        try:
            parameters = inspect.signature(clarify).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "reason" in parameters:
            return clarify(request.state, request.question, request.options, CLARIFY_REASON)
        return clarify(request.state, normalized, request.question, request.options, CLARIFY_REASON)

    @staticmethod
    def _has_new_evidence(clarified: dict[str, Any], baseline: dict[str, Any]) -> bool:
        return _canonical(clarified) != _canonical(baseline)

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
        policy = self.policy
        threshold = (
            request.confidence_threshold
            if request.confidence_threshold is not None
            else policy.confidence_threshold
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

        # Structured baseline for the same original input the first pass saw. A
        # clarifier that only restates it adds nothing and must not trigger a re-roll.
        if normalized is None:
            normalized = self.semantic.normalize(request.state, request.question, request.options)

        clarified = self.clarify_state(request, normalized)
        if not self._has_new_evidence(clarified, normalized):
            return self._response(
                first,
                threshold=threshold,
                mode="clarified",
                attempts=1,
                normalized_state=clarified,
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