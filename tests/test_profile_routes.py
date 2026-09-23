from unittest.mock import AsyncMock

import pytest

from app.api.routes import profile as profile_routes
from app.core.auth import require_user_id
from app.main import app
from app.services.profile import ProfileData, ProfileError


@pytest.fixture(autouse=True)
def _override_user():
    app.dependency_overrides[require_user_id] = lambda: "user-123"
    yield
    app.dependency_overrides.clear()


def _sample_data(**overrides) -> ProfileData:
    base = {
        "user_id": "user-123",
        "name": None,
        "weight_kg": None,
        "injury_notes": None,
        "field_sources": {},
    }
    base.update(overrides)
    return ProfileData(**base)


def test_get_profile_returns_defaults(monkeypatch):
    monkeypatch.setattr(
        profile_routes.profile_service, "get_or_create_profile", AsyncMock(return_value=_sample_data())
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/api/profile")

    assert response.status_code == 200
    body = response.json()
    assert body["field_sources"] == {}


def test_patch_profile_updates_and_returns_fields(monkeypatch):
    mock_update = AsyncMock(
        return_value=_sample_data(weight_kg=70.0, field_sources={"weight_kg": "user"})
    )
    monkeypatch.setattr(profile_routes.profile_service, "update_profile", mock_update)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.patch("/api/profile", json={"weight_kg": 70.0})

    assert response.status_code == 200
    assert response.json()["weight_kg"] == 70.0
    mock_update.assert_awaited_once_with(
        "user-123", {"weight_kg": 70.0}, source="user"
    )


def test_patch_profile_with_no_body_fields_does_not_call_update(monkeypatch):
    mock_update = AsyncMock()
    mock_get = AsyncMock(return_value=_sample_data())
    monkeypatch.setattr(profile_routes.profile_service, "update_profile", mock_update)
    monkeypatch.setattr(profile_routes.profile_service, "get_or_create_profile", mock_get)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.patch("/api/profile", json={})

    assert response.status_code == 200
    mock_update.assert_not_awaited()
    mock_get.assert_awaited_once()


def test_patch_profile_invalid_value_returns_422(monkeypatch):
    monkeypatch.setattr(
        profile_routes.profile_service,
        "update_profile",
        AsyncMock(side_effect=ProfileError("Unknown profile field(s): [...]")),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.patch("/api/profile", json={"name": "x"})

    assert response.status_code == 422


def test_patch_profile_can_explicitly_clear_a_field(monkeypatch):
    mock_update = AsyncMock(return_value=_sample_data(injury_notes=None, field_sources={"injury_notes": "user"}))
    monkeypatch.setattr(profile_routes.profile_service, "update_profile", mock_update)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.patch("/api/profile", json={"injury_notes": None})

    assert response.status_code == 200
    mock_update.assert_awaited_once_with("user-123", {"injury_notes": None}, source="user")


def test_profile_routes_require_sign_in_without_dependency_override():
    app.dependency_overrides.clear()
    from fastapi.testclient import TestClient

    client = TestClient(app)
    assert client.get("/api/profile").status_code == 401
    assert client.patch("/api/profile", json={"name": "x"}).status_code == 401
