from __future__ import annotations

import math
import os
import random
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

import pytest
from fastapi.testclient import TestClient

from tiny_decision_stack import api
from tiny_decision_stack.backends import BackendValidationError, ScoredChoice, validate_scored_choice
from tiny_decision_stack.models import DecisionRequest, DecisionResponse
from tiny_decision_stack.orchestrator import DecisionOrchestrator


SEED = int(os.getenv("TDS_SWARM_SEED", "0"))
CASES = int(os.getenv("TDS_SWARM_CASES", "250"))


class FakeSemantic:
    status = "ready"

    def normalize(self, state, question, options):
        return {
            "facts": [state],
            "constraints": [],
            "risks": [],
            "evidence_for": {},
            "evidence_against": {},
            "missing_information": [],
        }

    def clarify(self, state, normalized, question, options, probabilities):
        return {**normalized, "constraints": ["clarified"]}


class FixedDecider:
    status = "ready"

    def __init__(self, scored):
        self.scored = scored

    def decide(self, state, question, options):
        return self.scored


def _options(count: int) -> dict[str, str]:
    return {f"o{i}": f"option {i}" for i in range(count)}


def _valid_scored(rng: random.Random, options: dict[str, str]) -> ScoredChoice:
    keys = list(options)
    raw = [rng.random() + 1e-9 for _ in keys]
    total = sum(raw)
    probabilities = {key: value / total for key, value in zip(keys, raw)}
    choice = max(probabilities, key=probabilities.__getitem__)
    return ScoredChoice(choice=choice, confidence=probabilities[choice], probabilities=probabilities)


def _random_text(rng: random.Random, minimum: int = 1, maximum: int = 800) -> str:
    alphabet = string.ascii_letters + string.digits + " .,:;!?-_ café✅日本語العربية"
    size = rng.randint(minimum, maximum)
    value = "".join(rng.choice(alphabet) for _ in range(size)).strip()
    return value or "x"


def test_seeded_valid_decisions_preserve_invariants():
    rng = random.Random(SEED)
    semantic = FakeSemantic()
    for _ in range(CASES):
        option_count = rng.randint(2, 32)
        options = _options(option_count)
        scored = _valid_scored(rng, options)
        request = DecisionRequest(
            state=_random_text(rng),
            question="Choose the safest valid action",
            options=options,
            confidence_threshold=rng.random(),
            preprocess=rng.choice(["auto", "direct", "normalize"]),
            clarify_on_low_confidence=False,
        )
        result = DecisionOrchestrator(semantic, FixedDecider(scored), direct_max_chars=500).decide(request)
        assert result.choice is None or result.choice in options
        assert 0.0 <= result.confidence <= 1.0
        assert set(result.probabilities) == set(options)
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in result.probabilities.values())
        assert abs(sum(result.probabilities.values()) - 1.0) <= 0.02
        assert result.abstained == (scored.confidence < request.confidence_threshold)


def test_seeded_invalid_backend_outputs_always_fail_closed():
    rng = random.Random(SEED ^ 0x5EED)
    for _ in range(CASES):
        options = _options(rng.randint(2, 12))
        scored = _valid_scored(rng, options)
        mutation = rng.randrange(6)
        probabilities = dict(scored.probabilities)
        choice = scored.choice
        confidence = scored.confidence
        if mutation == 0:
            choice = "__unknown__"
        elif mutation == 1:
            confidence = float("nan")
        elif mutation == 2:
            probabilities.pop(next(iter(probabilities)))
        elif mutation == 3:
            probabilities[next(iter(probabilities))] = -0.1
        elif mutation == 4:
            probabilities = {key: value * 0.5 for key, value in probabilities.items()}
        else:
            confidence = min(1.0, scored.confidence + 0.25)
            if abs(confidence - probabilities[choice]) <= 0.02:
                confidence = max(0.0, scored.confidence - 0.25)
        with pytest.raises(BackendValidationError):
            validate_scored_choice(ScoredChoice(choice, confidence, probabilities), options)


def test_seeded_routing_is_deterministic_for_structured_and_ambiguous_inputs():
    rng = random.Random(SEED + 17)
    orchestrator = DecisionOrchestrator(FakeSemantic(), FixedDecider(ScoredChoice("a", 0.9, {"a": 0.9, "b": 0.1})))
    for _ in range(max(25, CASES // 5)):
        structured = DecisionRequest(
            state=f'{{"value": {rng.randint(0, 9999)}, "ok": true}}',
            question="q",
            options={"a": "A", "b": "B"},
            preprocess="auto",
        )
        assert orchestrator._should_normalize(structured) is False
        ambiguous = DecisionRequest(
            state=f"Signal {rng.randint(0,9999)} is valid but authorization is unclear.",
            question="q",
            options={"a": "A", "b": "B"},
            preprocess="auto",
        )
        assert orchestrator._should_normalize(ambiguous) is True


class BlockingOrchestrator:
    semantic = type("Backend", (), {"status": "ready"})()
    decider = type("Backend", (), {"status": "ready"})()

    def __init__(self, started: threading.Event, release: threading.Event):
        self.started = started
        self.release = release

    def decide(self, request):
        self.started.set()
        self.release.wait(timeout=2)
        return DecisionResponse(
            choice="a",
            confidence=0.9,
            probabilities={"a": 0.9, "b": 0.1},
            abstained=False,
            mode="direct",
            attempts=1,
        )


def _payload():
    return {"state": "x", "question": "q", "options": {"a": "A", "b": "B"}}


def test_api_backpressure_rejects_overcommit(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(api, "orchestrator", BlockingOrchestrator(started, release))
    monkeypatch.setattr(api, "_inference_slots", BoundedSemaphore(1))
    monkeypatch.setattr(api, "_executor", executor)
    monkeypatch.setattr(api, "INFERENCE_TIMEOUT_SECONDS", 2.0)
    client = TestClient(api.app)

    with ThreadPoolExecutor(max_workers=1) as callers:
        first = callers.submit(client.post, "/decide", json=_payload())
        assert started.wait(timeout=1)
        second = client.post("/decide", json=_payload())
        assert second.status_code == 503
        assert "busy" in second.json()["detail"]
        release.set()
        assert first.result(timeout=2).status_code == 200
    executor.shutdown(wait=True)


def test_timeout_keeps_capacity_reserved_until_worker_finishes(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(api, "orchestrator", BlockingOrchestrator(started, release))
    monkeypatch.setattr(api, "_inference_slots", BoundedSemaphore(1))
    monkeypatch.setattr(api, "_executor", executor)
    monkeypatch.setattr(api, "INFERENCE_TIMEOUT_SECONDS", 0.03)
    client = TestClient(api.app)

    first = client.post("/decide", json=_payload())
    assert first.status_code == 503
    assert "timed out" in first.json()["detail"]
    assert started.is_set()
    second = client.post("/decide", json=_payload())
    assert second.status_code == 503
    assert "busy" in second.json()["detail"]
    release.set()

    deadline = time.time() + 2
    while time.time() < deadline:
        third = client.post("/decide", json=_payload())
        if third.status_code == 200:
            break
        time.sleep(0.02)
    else:
        pytest.fail("inference capacity was not released after worker completion")
    executor.shutdown(wait=True)
