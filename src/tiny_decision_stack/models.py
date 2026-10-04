from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DecisionRequest(BaseModel):
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: dict[str, str]
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    preprocess: Literal["auto", "direct", "normalize"] = "auto"
    clarify_on_low_confidence: bool = True

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) < 2:
            raise ValueError("at least two options are required")
        if any(not key.strip() or not description.strip() for key, description in value.items()):
            raise ValueError("option labels and descriptions must be non-empty")
        return value


class DecisionResponse(BaseModel):
    choice: str | None
    confidence: float
    probabilities: dict[str, float]
    abstained: bool
    mode: Literal["direct", "normalized", "clarified"]
    attempts: int
    normalized_state: dict | None = None
