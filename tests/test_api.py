from fastapi.testclient import TestClient

from tiny_decision_stack import api
from tiny_decision_stack.backends import BackendError, ScoredChoice
from tiny_decision_stack.models import DecisionResponse


class ReadyBackend:
    status = "ready"


class FakeOrchestrator:
    semantic = ReadyBackend()
    decider = ReadyBackend()

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def decide(self, request):
        if self.error:
            raise self.error
        return self.result


def payload(**overrides):
    data = {
        "state": "duplicate charge",
        "question": "What should we do?",
        "options": {"refund": "refund duplicate", "escalate": "human review"},
    }
    data.update(overrides)
    return data


def test_health_endpoints(monkeypatch):
    monkeypatch.setattr(api, "orchestrator", FakeOrchestrator())
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready").json()
    assert ready["status"] == "ready"
    assert ready["lazy_loading"] is True


def test_decide_success(monkeypatch):
    result = DecisionResponse(
        choice="refund",
        confidence=0.9,
        probabilities={"refund": 0.9, "escalate": 0.1},
        abstained=False,
        mode="direct",
        attempts=1,
    )
    monkeypatch.setattr(api, "orchestrator", FakeOrchestrator(result=result))
    client = TestClient(api.app)
    response = client.post("/decide", json=payload())
    assert response.status_code == 200
    assert response.json()["choice"] == "refund"


def test_backend_failure_returns_503(monkeypatch):
    monkeypatch.setattr(api, "orchestrator", FakeOrchestrator(error=BackendError("model unavailable")))
    client = TestClient(api.app)
    response = client.post("/decide", json=payload())
    assert response.status_code == 503
    assert "model unavailable" in response.json()["detail"]


def test_state_and_option_limits(monkeypatch):
    monkeypatch.setattr(api, "orchestrator", FakeOrchestrator())
    monkeypatch.setattr(api, "MAX_STATE_CHARS", 5)
    client = TestClient(api.app)
    assert client.post("/decide", json=payload(state="123456")).status_code == 422

    monkeypatch.setattr(api, "MAX_STATE_CHARS", 1000)
    monkeypatch.setattr(api, "MAX_OPTIONS", 2)
    options = {"a": "A", "b": "B", "c": "C"}
    assert client.post("/decide", json=payload(options=options)).status_code == 422


def test_unicode_request_is_accepted(monkeypatch):
    result = DecisionResponse(
        choice=None,
        confidence=0.5,
        probabilities={"sim": 0.5, "não": 0.5},
        abstained=True,
        mode="direct",
        attempts=1,
    )
    monkeypatch.setattr(api, "orchestrator", FakeOrchestrator(result=result))
    client = TestClient(api.app)
    response = client.post(
        "/decide",
        json=payload(state="café ✅", options={"sim": "ação", "não": "revisão"}),
    )
    assert response.status_code == 200
