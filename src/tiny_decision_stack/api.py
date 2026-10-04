from __future__ import annotations

import os

from fastapi import FastAPI

from .backends import LocalLFMBackend, LocalOpenDeciderBackend
from .models import DecisionRequest, DecisionResponse
from .orchestrator import DecisionOrchestrator


LFM_MODEL = os.getenv("LFM_MODEL", "LiquidAI/LFM2.5-350M")
OPENDECIDER_MODEL = os.getenv("OPENDECIDER_MODEL", "manjunathshiva/opendecider-nano")
MODEL_DEVICE = os.getenv("MODEL_DEVICE") or None
DIRECT_MAX_CHARS = int(os.getenv("DIRECT_MAX_CHARS", "500"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.75"))

orchestrator = DecisionOrchestrator(
    semantic=LocalLFMBackend(LFM_MODEL, MODEL_DEVICE),
    decider=LocalOpenDeciderBackend(OPENDECIDER_MODEL, MODEL_DEVICE),
    default_confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    direct_max_chars=DIRECT_MAX_CHARS,
)

app = FastAPI(
    title="Tiny Decision Stack",
    version="0.1.0",
    description="LFM2.5-350M semantic preprocessing + OpenDecider-nano calibrated decisions",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/decide", response_model=DecisionResponse)
def decide(request: DecisionRequest) -> DecisionResponse:
    return orchestrator.decide(request)
