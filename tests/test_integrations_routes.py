from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.api.routes import integrations as integrations_routes
from app.core.auth import require_user_id
from app.main import app
from app.services.coros_oauth import ConnectionStatus, CorosOAuthError
from app.services.coros_sync import SyncStatus


@pytest.fixture(autouse=True)
def _override_user():
    app.dependency_overrides[require_user_id] = lambda: "user-123"
    yield
    app.dependency_overrides.clear()


def test_connect_returns_authorization_url(monkeypatch):
    monkeypatch.setattr(
        integrations_routes.coros_oauth,
        "build_authorization_url",
        AsyncMock(return_value="https://mcpus.coros.com/oauth2/authorize?state=abc"),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/api/integrations/coros/connect")

    assert response.status_code == 200
    assert response.json() == {"authorization_url": "https://mcpus.coros.com/oauth2/authorize?state=abc"}


def test_connect_failure_returns_502(monkeypatch):
    monkeypatch.setattr(
        integrations_routes.coros_oauth,
        "build_authorization_url",
        AsyncMock(side_effect=CorosOAuthError('Token endpoint returned 400: {"error":"secret-ish"}')),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/api/integrations/coros/connect")

    assert response.status_code == 502
    # COROS's raw response body stays in the server log, never in the client-facing detail.
    assert response.json() == {"detail": "Couldn't start the COROS connection. Please try again."}


def test_callback_success_redirects_to_profile_connected(monkeypatch):
    monkeypatch.setattr(
        integrations_routes.coros_oauth,
        "handle_callback",
        AsyncMock(return_value="user-123"),
    )
    backfill = AsyncMock()
    monkeypatch.setattr(integrations_routes.coros_sync, "backfill_user", backfill)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get(
        "/api/integrations/coros/callback",
        params={"code": "abc", "state": "xyz"},
        follow_redirects=False,
    )

    assert response.status_code in (302, 307)
    assert response.headers["location"].endswith("/profile?coros=connected")
    backfill.assert_awaited_once_with("user-123")


def test_callback_failure_redirects_to_profile_error(monkeypatch):
    monkeypatch.setattr(
        integrations_routes.coros_oauth,
        "handle_callback",
        AsyncMock(side_effect=CorosOAuthError("bad state")),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get(
        "/api/integrations/coros/callback",
        params={"code": "abc", "state": "xyz"},
        follow_redirects=False,
    )

    assert response.status_code in (302, 307)
    assert response.headers["location"].endswith("/profile?coros=error")


def test_status_not_connected(monkeypatch):
    monkeypatch.setattr(
        integrations_routes.coros_oauth,
        "get_status",
        AsyncMock(return_value=ConnectionStatus(connected=False)),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/api/integrations/coros/status")

    assert response.status_code == 200
    assert response.json() == {"connected": False, "connected_at": None, "syncing": False, "last_synced_at": None}


def test_status_connected(monkeypatch):
    connected_at = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(
        integrations_routes.coros_oauth,
        "get_status",
        AsyncMock(return_value=ConnectionStatus(connected=True, connected_at=connected_at)),
    )
    monkeypatch.setattr(
        integrations_routes.coros_sync,
        "get_sync_status",
        AsyncMock(return_value=None),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/api/integrations/coros/status")

    assert response.status_code == 200
    assert response.json() == {
        "connected": True,
        "connected_at": "2026-01-01T00:00:00Z",
        "syncing": True,
        "last_synced_at": None,
    }


def test_status_connected_and_synced(monkeypatch):
    monkeypatch.setattr(
        integrations_routes.coros_oauth,
        "get_status",
        AsyncMock(return_value=ConnectionStatus(connected=True, connected_at=datetime(2026, 1, 1, tzinfo=UTC))),
    )
    monkeypatch.setattr(
        integrations_routes.coros_sync,
        "get_sync_status",
        AsyncMock(
            return_value=SyncStatus(
                status="ok",
                last_synced_at=datetime(2026, 1, 3, 6, 30, tzinfo=UTC),
                last_backfill_completed_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
                last_error=None,
            )
        ),
    )
    from fastapi.testclient import TestClient

    response = TestClient(app).get("/api/integrations/coros/status")

    assert response.status_code == 200
    body = response.json()
    assert body["syncing"] is False
    assert body["last_synced_at"] == "2026-01-03T06:30:00Z"


def test_disconnect_calls_service(monkeypatch):
    mock_disconnect = AsyncMock()
    monkeypatch.setattr(integrations_routes.coros_oauth, "disconnect", mock_disconnect)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/api/integrations/coros/disconnect")

    assert response.status_code == 200
    assert response.json() == {"status": "disconnected"}
    mock_disconnect.assert_awaited_once_with("user-123")


def test_disconnect_failure_returns_502(monkeypatch):
    monkeypatch.setattr(
        integrations_routes.coros_oauth,
        "disconnect",
        AsyncMock(side_effect=CorosOAuthError('Revoke returned 500: {"raw": "body"}')),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/api/integrations/coros/disconnect")

    assert response.status_code == 502
    assert response.json() == {"detail": "Couldn't disconnect COROS. Please try again."}


def test_coros_routes_require_sign_in_without_dependency_override():
    # Regression test: with the real require_user_id dependency (no
    # override), anonymous callers must be rejected — COROS OAuth tokens
    # need a stable account to be held against.
    app.dependency_overrides.clear()
    from fastapi.testclient import TestClient

    client = TestClient(app)
    assert client.post("/api/integrations/coros/connect").status_code == 401
    assert client.get("/api/integrations/coros/status").status_code == 401
    assert client.post("/api/integrations/coros/disconnect").status_code == 401
