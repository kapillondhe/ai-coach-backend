from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.agents import coach_agent
from app.agents.coach_agent import get_coach_agent
from app.api.routes import coach as coach_routes
from app.core import rate_limit
from app.core.auth import get_current_user_id
from app.core.config import get_settings
from app.core.rate_limit import SlidingWindowLimiter
from app.main import app


class _FakeResult:
    def __init__(self, output) -> None:
        self.output = output


class _FakeAgent:
    async def run(self, message: str, message_history=None) -> _FakeResult:
        return _FakeResult(output=coach_agent.CoachReply(reply="ok", suggestions=[]))


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(coach_routes.scope_service, "is_off_topic", AsyncMock(return_value=False))
    rate_limit.chat_limiter.reset()
    app.dependency_overrides[get_coach_agent] = lambda: _FakeAgent()
    yield
    app.dependency_overrides.clear()
    rate_limit.chat_limiter.reset()
    get_settings.cache_clear()


def _set_limit(monkeypatch, limit: int) -> None:
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", str(limit))
    get_settings.cache_clear()


def _post(client: TestClient, path: str = "/api/coach/chat", **headers: str):
    return client.post(path, json={"message": "hello"}, headers=headers)


def test_chat_returns_429_with_retry_after_once_limit_exceeded(monkeypatch):
    _set_limit(monkeypatch, 3)
    client = TestClient(app)

    assert [_post(client).status_code for _ in range(3)] == [200, 200, 200]
    response = _post(client)

    assert response.status_code == 429
    assert 1 <= int(response.headers["Retry-After"]) <= 60


def test_chat_and_stream_share_one_bucket(monkeypatch):
    _set_limit(monkeypatch, 1)
    client = TestClient(app)

    assert _post(client).status_code == 200
    assert _post(client, "/api/coach/chat/stream").status_code == 429


def test_cross_origin_429_exposes_retry_after_to_the_browser(monkeypatch):
    # The frontend is on another origin; the browser only lets JS read Retry-After
    # if CORS lists it in Access-Control-Expose-Headers.
    _set_limit(monkeypatch, 1)
    client = TestClient(app)
    origin = get_settings().cors_origin_list[0]

    _post(client, Origin=origin)
    response = _post(client, Origin=origin)

    assert response.status_code == 429
    assert response.headers["access-control-allow-origin"] == origin
    assert "retry-after" in response.headers["access-control-expose-headers"].lower()


def test_rate_limit_is_per_forwarded_ip(monkeypatch):
    _set_limit(monkeypatch, 1)
    client = TestClient(app)

    assert _post(client, **{"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}).status_code == 200
    # Same left-most client, different proxy hop: same bucket.
    assert _post(client, **{"X-Forwarded-For": "1.1.1.1, 10.0.0.2"}).status_code == 429
    assert _post(client, **{"X-Forwarded-For": "2.2.2.2, 10.0.0.1"}).status_code == 200


def test_rate_limit_falls_back_to_client_host_without_forwarded_for(monkeypatch):
    _set_limit(monkeypatch, 1)

    assert _post(TestClient(app, client=("3.3.3.3", 1))).status_code == 200
    assert _post(TestClient(app, client=("3.3.3.3", 2))).status_code == 429
    assert _post(TestClient(app, client=("4.4.4.4", 1))).status_code == 200


def test_rate_limit_is_per_signed_in_user_regardless_of_ip(monkeypatch):
    from app.services import conversation as conversation_service

    _set_limit(monkeypatch, 1)
    monkeypatch.setattr(conversation_service, "create_conversation", AsyncMock(return_value="conv-new"))
    monkeypatch.setattr(conversation_service, "append_messages", AsyncMock())
    monkeypatch.setattr(coach_routes.titling_service, "generate_and_set_title", AsyncMock())
    client = TestClient(app)

    app.dependency_overrides[get_current_user_id] = lambda: "user-a"
    assert _post(client, **{"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert _post(client, **{"X-Forwarded-For": "9.9.9.9"}).status_code == 429

    app.dependency_overrides[get_current_user_id] = lambda: "user-b"
    assert _post(client, **{"X-Forwarded-For": "1.1.1.1"}).status_code == 200


def test_rate_limit_zero_disables(monkeypatch):
    _set_limit(monkeypatch, 0)
    client = TestClient(app)

    assert {_post(client).status_code for _ in range(30)} == {200}
    assert len(rate_limit.chat_limiter) == 0


def test_rejected_request_does_no_expensive_work(monkeypatch):
    """The limit runs before the context load and agent setup, and starts neither."""
    from app.services import conversation as conversation_service

    _set_limit(monkeypatch, 1)
    create_mock = AsyncMock(return_value="conv-new")
    monkeypatch.setattr(conversation_service, "create_conversation", create_mock)
    monkeypatch.setattr(conversation_service, "append_messages", AsyncMock())
    monkeypatch.setattr(coach_routes.titling_service, "generate_and_set_title", AsyncMock())
    agent_builds: list[int] = []

    def _agent():
        agent_builds.append(1)
        return _FakeAgent()

    app.dependency_overrides[get_coach_agent] = _agent
    app.dependency_overrides[get_current_user_id] = lambda: "user-a"
    client = TestClient(app)

    assert _post(client).status_code == 200
    assert _post(client).status_code == 429
    assert create_mock.await_count == 1
    assert len(agent_builds) == 1


def test_limiter_window_slides_and_reports_retry_after():
    limiter = SlidingWindowLimiter(window_seconds=60)

    assert limiter.hit("k", 2, now=0.0) is None
    assert limiter.hit("k", 2, now=10.0) is None
    assert limiter.hit("k", 2, now=20.0) == pytest.approx(40.0)
    assert limiter.hit("k", 2, now=60.5) is None  # the t=0 hit has aged out


def test_limiter_memory_is_bounded():
    limiter = SlidingWindowLimiter(window_seconds=60, max_keys=3)

    for i in range(10):
        limiter.hit(f"k{i}", 5, now=float(i))
    assert len(limiter) == 3

    # Idle keys are swept once the window has passed.
    limiter.hit("fresh", 5, now=200.0)
    assert len(limiter) == 1
