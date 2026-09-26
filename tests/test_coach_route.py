from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.agents.coach_agent import get_coach_agent
from app.api.routes import coach as coach_routes
from app.core.auth import get_current_user_id
from app.main import app
from app.services.chat_opener import ChatOpener


def test_opener_anonymous(monkeypatch):
    monkeypatch.setattr(
        coach_routes.chat_opener_service,
        "get_chat_opener",
        AsyncMock(return_value=ChatOpener(text="What are you training for?", chips=["a", "b"])),
    )
    client = TestClient(app)
    response = client.get("/api/coach/opener")

    assert response.status_code == 200
    assert response.json() == {"text": "What are you training for?", "chips": ["a", "b"]}


def test_opener_signed_in_passes_user_id(monkeypatch):
    mock_get_opener = AsyncMock(return_value=ChatOpener(text="Nice ride.", chips=[]))
    monkeypatch.setattr(coach_routes.chat_opener_service, "get_chat_opener", mock_get_opener)
    app.dependency_overrides[get_current_user_id] = lambda: "user-123"
    try:
        client = TestClient(app)
        response = client.get("/api/coach/opener")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["text"] == "Nice ride."
    mock_get_opener.assert_awaited_once_with("user-123")


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
    async def run(self, message: str, message_history=None) -> _FakeResult:
        return _FakeResult(output=f"echo: {message}")

    def run_stream(self, message: str, message_history=None) -> _FakeStreamContext:
        return _FakeStreamContext([f"echo: {message}"])


class _FakeFailingAgent:
    async def run(self, message: str, message_history=None) -> _FakeResult:
        raise RuntimeError("boom")

    def run_stream(self, message: str, message_history=None):
        raise RuntimeError("boom")


def test_chat_returns_agent_reply():
    app.dependency_overrides[get_coach_agent] = lambda: _FakeAgent()
    try:
        client = TestClient(app)
        response = client.post("/api/coach/chat", json={"message": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: hello", "conversation_id": None}


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


class _RecordingAgent:
    """Captures the message_history it was called with, for assertion."""

    def __init__(self) -> None:
        self.received_history: list = []

    async def run(self, message: str, message_history=None) -> _FakeResult:
        self.received_history = list(message_history or [])
        return _FakeResult(output=f"echo: {message}")


def test_chat_passes_prior_turns_as_message_history():
    recording_agent = _RecordingAgent()
    app.dependency_overrides[get_coach_agent] = lambda: recording_agent
    try:
        client = TestClient(app)
        response = client.post(
            "/api/coach/chat",
            json={
                "message": "what's my name?",
                "history": [
                    {"role": "user", "content": "my name is Alex"},
                    {"role": "assistant", "content": "Nice to meet you, Alex!"},
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(recording_agent.received_history) == 2
    assert recording_agent.received_history[0].parts[0].content == "my name is Alex"
    assert recording_agent.received_history[1].parts[0].content == "Nice to meet you, Alex!"


def test_chat_with_no_history_passes_empty_list():
    recording_agent = _RecordingAgent()
    app.dependency_overrides[get_coach_agent] = lambda: recording_agent
    try:
        client = TestClient(app)
        response = client.post("/api/coach/chat", json={"message": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert recording_agent.received_history == []


class _FakeConversationData:
    def __init__(self, id: str, messages: list[tuple[str, str]]) -> None:
        from app.services.conversation import MessageData

        self.id = id
        self.user_id = "user-123"
        self.title = None
        self.messages = [MessageData(role=r, content=c) for r, c in messages]


def test_signed_in_chat_with_no_conversation_id_creates_one(monkeypatch):
    from app.services import conversation as conversation_service

    recording_agent = _RecordingAgent()
    monkeypatch.setattr(
        conversation_service, "create_conversation", AsyncMock(return_value="conv-new")
    )
    append_mock = AsyncMock()
    monkeypatch.setattr(conversation_service, "append_messages", append_mock)

    app.dependency_overrides[get_coach_agent] = lambda: recording_agent
    app.dependency_overrides[get_current_user_id] = lambda: "user-123"
    try:
        client = TestClient(app)
        response = client.post("/api/coach/chat", json={"message": "hi there"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "conv-new"
    assert recording_agent.received_history == []
    append_mock.assert_awaited_once()
    args = append_mock.await_args.args
    assert args[0] == "conv-new"
    assert args[1] == "user-123"


def test_signed_in_chat_with_existing_conversation_id_loads_history(monkeypatch):
    from app.services import conversation as conversation_service

    recording_agent = _RecordingAgent()
    fake_data = _FakeConversationData("conv-1", [("user", "my name is Alex"), ("assistant", "hi Alex")])
    monkeypatch.setattr(
        conversation_service, "get_conversation", AsyncMock(return_value=fake_data)
    )
    monkeypatch.setattr(conversation_service, "append_messages", AsyncMock())

    app.dependency_overrides[get_coach_agent] = lambda: recording_agent
    app.dependency_overrides[get_current_user_id] = lambda: "user-123"
    try:
        client = TestClient(app)
        response = client.post(
            "/api/coach/chat", json={"message": "what's my name?", "conversation_id": "conv-1"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "conv-1"
    assert len(recording_agent.received_history) == 2
    assert recording_agent.received_history[0].parts[0].content == "my name is Alex"


def test_signed_in_chat_with_unknown_conversation_id_returns_404(monkeypatch):
    from app.services import conversation as conversation_service
    from app.services.conversation import ConversationNotFoundError

    monkeypatch.setattr(
        conversation_service,
        "get_conversation",
        AsyncMock(side_effect=ConversationNotFoundError("conv-missing")),
    )

    app.dependency_overrides[get_coach_agent] = lambda: _RecordingAgent()
    app.dependency_overrides[get_current_user_id] = lambda: "user-123"
    try:
        client = TestClient(app)
        response = client.post(
            "/api/coach/chat", json={"message": "hi", "conversation_id": "conv-missing"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_signed_in_chat_ignores_client_supplied_history(monkeypatch):
    """Signed-in requests use server-side conversation state, not the anonymous `history` field."""
    from app.services import conversation as conversation_service

    recording_agent = _RecordingAgent()
    monkeypatch.setattr(
        conversation_service, "create_conversation", AsyncMock(return_value="conv-new")
    )
    monkeypatch.setattr(conversation_service, "append_messages", AsyncMock())

    app.dependency_overrides[get_coach_agent] = lambda: recording_agent
    app.dependency_overrides[get_current_user_id] = lambda: "user-123"
    try:
        client = TestClient(app)
        response = client.post(
            "/api/coach/chat",
            json={
                "message": "hello",
                "history": [{"role": "user", "content": "ignored, client-side only"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert recording_agent.received_history == []
