from fastapi.testclient import TestClient

from app.agents.coach_agent import get_coach_agent
from app.main import app


class _FakeResult:
    def __init__(self, output: str) -> None:
        self.output = output


class _FakeStreamedResult:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def stream_text(self, delta: bool = False):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContext:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeStreamedResult:
        return _FakeStreamedResult(self._chunks)

    async def __aexit__(self, *exc_info) -> None:
        return None


class _FakeAgent:
    async def run(self, message: str) -> _FakeResult:
        return _FakeResult(output=f"echo: {message}")

    def run_stream(self, message: str) -> _FakeStreamContext:
        return _FakeStreamContext([f"echo: {message}"])


class _FakeFailingAgent:
    async def run(self, message: str) -> _FakeResult:
        raise RuntimeError("boom")

    def run_stream(self, message: str):
        raise RuntimeError("boom")


def test_chat_returns_agent_reply():
    app.dependency_overrides[get_coach_agent] = lambda: _FakeAgent()
    try:
        client = TestClient(app)
        response = client.post("/api/coach/chat", json={"message": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: hello"}


def test_chat_stream_returns_sse_chunks():
    app.dependency_overrides[get_coach_agent] = lambda: _FakeAgent()
    try:
        client = TestClient(app)
        response = client.post("/api/coach/chat/stream", json={"message": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"delta": "echo: hello"}' in response.text
    assert "event: done" in response.text


def test_chat_stream_emits_error_event_on_failure():
    app.dependency_overrides[get_coach_agent] = lambda: _FakeFailingAgent()
    try:
        client = TestClient(app)
        response = client.post("/api/coach/chat/stream", json={"message": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "event: error" in response.text
