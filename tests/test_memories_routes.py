from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.auth import require_user_id
from app.main import app
from app.services import memory as memory_service
from app.services.memory import MemoryData


@pytest.fixture(autouse=True)
def _override_user():
    app.dependency_overrides[require_user_id] = lambda: "user-123"
    yield
    app.dependency_overrides.clear()


def test_list_memories_returns_stored_facts(monkeypatch):
    memories = [
        MemoryData(id="mem-1", content="Training for a first 70.3", created_at=datetime(2026, 1, 1, tzinfo=UTC)),
    ]
    monkeypatch.setattr(memory_service, "list_memories", AsyncMock(return_value=memories))

    client = TestClient(app)
    response = client.get("/api/memories")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == "mem-1"
    assert body[0]["content"] == "Training for a first 70.3"


def test_delete_memory_returns_success(monkeypatch):
    monkeypatch.setattr(memory_service, "forget", AsyncMock(return_value=True))

    client = TestClient(app)
    response = client.delete("/api/memories/mem-1")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}


def test_delete_memory_returns_404_when_not_owned(monkeypatch):
    monkeypatch.setattr(memory_service, "forget", AsyncMock(return_value=False))

    client = TestClient(app)
    response = client.delete("/api/memories/mem-1")

    assert response.status_code == 404


def test_memories_routes_require_sign_in_without_dependency_override():
    app.dependency_overrides.clear()
    client = TestClient(app)

    assert client.get("/api/memories").status_code == 401
    assert client.delete("/api/memories/mem-1").status_code == 401
