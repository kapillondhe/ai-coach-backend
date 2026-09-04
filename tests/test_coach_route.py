from fastapi.testclient import TestClient

from app.agents.coach_agent import get_coach_agent
from app.main import app


class _FakeResult:
    def __init__(self, output: str) -> None:
        self.output = output


class _FakeAgent:
    async def run(self, message: str) -> _FakeResult:
        return _FakeResult(output=f"echo: {message}")


def test_chat_returns_agent_reply():
    app.dependency_overrides[get_coach_agent] = lambda: _FakeAgent()
    try:
        client = TestClient(app)
        response = client.post("/api/coach/chat", json={"message": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: hello"}
