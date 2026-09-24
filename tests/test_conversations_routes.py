from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.auth import require_user_id
from app.main import app
from app.services import conversation as conversation_service
from app.services.conversation import ConversationData, ConversationNotFoundError, ConversationSummary, MessageData


@pytest.fixture(autouse=True)
def _override_user():
    app.dependency_overrides[require_user_id] = lambda: "user-123"
    yield
    app.dependency_overrides.clear()


def test_list_conversations_returns_summaries(monkeypatch):
    summaries = [
        ConversationSummary(id="conv-1", title="Marathon plan", updated_at=datetime(2026, 1, 1, tzinfo=UTC)),
    ]
    monkeypatch.setattr(conversation_service, "list_conversations", AsyncMock(return_value=summaries))

    client = TestClient(app)
    response = client.get("/api/conversations")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == "conv-1"
    assert body[0]["title"] == "Marathon plan"


def test_get_conversation_returns_messages(monkeypatch):
    data = ConversationData(
        id="conv-1",
        user_id="user-123",
        title=None,
        messages=[MessageData(role="user", content="hi"), MessageData(role="assistant", content="hello!")],
    )
    monkeypatch.setattr(conversation_service, "get_conversation", AsyncMock(return_value=data))

    client = TestClient(app)
    response = client.get("/api/conversations/conv-1")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "conv-1"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]


def test_get_conversation_returns_404_when_not_owned(monkeypatch):
    monkeypatch.setattr(
        conversation_service,
        "get_conversation",
        AsyncMock(side_effect=ConversationNotFoundError("conv-1")),
    )

    client = TestClient(app)
    response = client.get("/api/conversations/conv-1")

    assert response.status_code == 404


def test_conversations_routes_require_sign_in_without_dependency_override():
    app.dependency_overrides.clear()
    client = TestClient(app)

    assert client.get("/api/conversations").status_code == 401
    assert client.get("/api/conversations/conv-1").status_code == 401
