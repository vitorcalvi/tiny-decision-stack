from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException

from .backends import BackendError, BackendValidationError, LocalLFMBackend, LocalOpenDeciderBackend
from .models import DecisionRequest, DecisionResponse, ErrorResponse
from .orchestrator import DecisionOrchestrator


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


LFM_MODEL = os.getenv("LFM_MODEL", "LiquidAI/LFM2.5-350M")
OPENDECIDER_MODEL = os.getenv("OPENDECIDER_MODEL", "manjunathshiva/opendecider-nano")
MODEL_DEVICE = os.getenv("MODEL_DEVICE") or None
DIRECT_MAX_CHARS = _env_int("DIRECT_MAX_CHARS", 500)
DEFAULT_CONFIDENCE_THRESHOLD = _env_float("DEFAULT_CONFIDENCE_THRESHOLD", 0.75, 0.0, 1.0)
MAX_STATE_CHARS = _env_int("MAX_STATE_CHARS", 20_000)
MAX_OPTIONS = _env_int("MAX_OPTIONS", 32, minimum=2)
MAX_CONCURRENT_INFERENCE = _env_int("MAX_CONCURRENT_INFERENCE", 1)
INFERENCE_TIMEOUT_SECONDS = _env_float("INFERENCE_TIMEOUT_SECONDS", 120.0, 0.1, 3600.0)

semantic_backend = LocalLFMBackend(LFM_MODEL, MODEL_DEVICE)
decision_backend = LocalOpenDeciderBackend(OPENDECIDER_MODEL, MODEL_DEVICE)
orchestrator = DecisionOrchestrator(
    semantic=semantic_backend,
    decider=decision_backend,
    default_confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    direct_max_chars=DIRECT_MAX_CHARS,
)
_inference_slots = BoundedSemaphore(MAX_CONCURRENT_INFERENCE)
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_INFERENCE, thread_name_prefix="decision-inference")

app = FastAPI(
    title="Tiny Decision Stack",
    version="0.1.1",
    description="Experimental local decision stack with semantic preprocessing, confidence gating, and abstention",
)


@app.get("/health")
@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready() -> dict[str, object]:
    semantic = getattr(orchestrator.semantic, "status", "unknown")
    decider = getattr(orchestrator.decider, "status", "unknown")
    ready = semantic == "ready" and decider == "ready"
    return {
        "status": "ready" if ready else "not_ready",
        "semantic_backend": semantic,
        "decision_backend": decider,
        "lazy_loading": True,
    }


@app.post(
    "/decide",
    response_model=DecisionResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def decide(request: DecisionRequest) -> DecisionResponse:
    if len(request.state) > MAX_STATE_CHARS:
        raise HTTPException(status_code=422, detail=f"state exceeds MAX_STATE_CHARS={MAX_STATE_CHARS}")
    if len(request.options) > MAX_OPTIONS:
        raise HTTPException(status_code=422, detail=f"options exceed MAX_OPTIONS={MAX_OPTIONS}")
    if not _inference_slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="inference capacity is busy")

    try:
        future = _executor.submit(orchestrator.decide, request)
    except Exception:
        _inference_slots.release()
        raise

    # Python cannot safely kill a running model call. The slot is therefore
    # released only when the worker actually finishes, even if the HTTP caller
    # already received a timeout. This prevents timeout-driven overcommit/OOM.
    future.add_done_callback(lambda _: _inference_slots.release())
    try:
        return future.result(timeout=INFERENCE_TIMEOUT_SECONDS)
    except FutureTimeoutError as exc:
        raise HTTPException(status_code=503, detail="inference timed out") from exc
    except BackendValidationError as exc:
        raise HTTPException(status_code=503, detail=f"backend output rejected: {exc}") from exc
    except BackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
