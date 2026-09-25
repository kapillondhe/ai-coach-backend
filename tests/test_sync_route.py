from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.routes import sync as sync_routes
from app.main import app


def test_run_sync_rejects_missing_secret(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("INTERNAL_SYNC_SECRET", "correct-secret")
    get_settings.cache_clear()

    client = TestClient(app)
    response = client.post("/api/internal/sync/run")

    assert response.status_code == 401
    get_settings.cache_clear()


def test_run_sync_rejects_wrong_secret(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("INTERNAL_SYNC_SECRET", "correct-secret")
    get_settings.cache_clear()

    client = TestClient(app)
    response = client.post("/api/internal/sync/run", headers={"X-Internal-Sync-Secret": "wrong"})

    assert response.status_code == 401
    get_settings.cache_clear()


def test_run_sync_returns_503_when_not_configured(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.delenv("INTERNAL_SYNC_SECRET", raising=False)
    get_settings.cache_clear()

    client = TestClient(app)
    response = client.post("/api/internal/sync/run", headers={"X-Internal-Sync-Secret": "anything"})

    assert response.status_code == 503
    get_settings.cache_clear()


def test_run_sync_succeeds_with_correct_secret(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("INTERNAL_SYNC_SECRET", "correct-secret")
    get_settings.cache_clear()
    monkeypatch.setattr(
        sync_routes.coros_sync, "run_periodic_sync", AsyncMock(return_value={"synced": 3, "failed": 1})
    )

    client = TestClient(app)
    response = client.post("/api/internal/sync/run", headers={"X-Internal-Sync-Secret": "correct-secret"})

    assert response.status_code == 200
    assert response.json() == {"synced": 3, "failed": 1}
    get_settings.cache_clear()
