from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.dml import Delete, Insert, Update

from app.services import coros_oauth


class _FakeSession:
    """Minimal SQLAlchemy AsyncSession stand-in: records executed statements, returns scripted rows."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.executed = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        row = self.results.pop(0) if self.results else None
        if isinstance(row, dict):
            row = SimpleNamespace(**row)
        return SimpleNamespace(first=lambda: row)

    def add(self, obj):
        self.executed.append(obj)

    async def commit(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _writes(session: _FakeSession) -> list:
    """The subset of executed statements that mutate data (inserts/updates/deletes)."""
    return [s for s in session.executed if isinstance(s, (Insert, Update, Delete))]


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_get_or_register_client_reuses_existing_row(monkeypatch):
    session = _FakeSession(results=[{"client_id": "existing-id", "client_secret": None}])
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    client_id, client_secret = await coros_oauth._get_or_register_client()

    assert client_id == "existing-id"
    assert client_secret is None
    assert _writes(session) == []  # no registration call needed


@pytest.mark.asyncio
async def test_get_or_register_client_registers_when_absent(monkeypatch):
    session = _FakeSession(results=[None])
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    fake_response = MagicMock()
    fake_response.status_code = 201
    fake_response.json.return_value = {"client_id": "new-id", "client_secret": "new-secret"}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = fake_response
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        client_id, client_secret = await coros_oauth._get_or_register_client()

    assert client_id == "new-id"
    assert client_secret == "new-secret"
    assert len(_writes(session)) == 1  # persisted the newly-registered client


@pytest.mark.asyncio
async def test_handle_callback_rejects_unknown_state(monkeypatch):
    session = _FakeSession(results=[None])  # delete...returning finds nothing
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    with pytest.raises(coros_oauth.CorosOAuthError, match="Unknown or already-used"):
        await coros_oauth.handle_callback(code="abc", state="does-not-exist")


@pytest.mark.asyncio
async def test_handle_callback_rejects_expired_state(monkeypatch):
    session = _FakeSession(
        results=[
            {
                "user_id": "user-123",
                "code_verifier": "verifier",
                "expires_at": datetime.now(UTC) - timedelta(minutes=1),
            }
        ]
    )
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    with pytest.raises(coros_oauth.CorosOAuthError, match="expired"):
        await coros_oauth.handle_callback(code="abc", state="stale")


@pytest.mark.asyncio
async def test_get_status_reports_not_connected(monkeypatch):
    session = _FakeSession(results=[None])
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    status = await coros_oauth.get_status("user-123")

    assert status.connected is False


@pytest.mark.asyncio
async def test_get_access_token_returns_none_when_not_connected(monkeypatch):
    session = _FakeSession(results=[None])
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    token = await coros_oauth.get_access_token("user-123")

    assert token is None


@pytest.mark.asyncio
async def test_disconnect_deletes_local_row_even_if_revocation_fails(monkeypatch):
    session = _FakeSession(
        results=[
            {"access_token_encrypted": "enc"},  # user_integrations lookup
            {"client_id": "id", "client_secret": None},  # _get_or_register_client
        ]
    )
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)
    monkeypatch.setattr(coros_oauth, "decrypt_token", lambda _: "plaintext-token")

    import httpx

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("unreachable")
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        await coros_oauth.disconnect("user-123")

    delete_calls = [s for s in _writes(session) if isinstance(s, Delete) and s.table.name == "user_integrations"]
    assert len(delete_calls) == 1


@pytest.mark.asyncio
async def test_disconnect_raises_coros_oauth_error_on_db_failure(monkeypatch):
    class _FailingCommitSession(_FakeSession):
        async def commit(self):
            raise SQLAlchemyError("connection lost")

    session = _FailingCommitSession(results=[None])  # no active integration -> revocation call is skipped
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    with pytest.raises(coros_oauth.CorosOAuthError, match="Failed to delete local COROS integration"):
        await coros_oauth.disconnect("user-123")
