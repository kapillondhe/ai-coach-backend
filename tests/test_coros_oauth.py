import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.dml import Delete, Insert, Update

from app.services import coros_oauth
from tests.fakes import FakeSession, ScriptedSession, row_result


def _writes(session: FakeSession) -> list:
    """The subset of executed statements that mutate data (inserts/updates/deletes)."""
    return [s for s in session.executed if isinstance(s, (Insert, Update, Delete))]


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.core.config import get_settings

    get_settings.cache_clear()
    coros_oauth._not_connected_cache.clear()
    coros_oauth._refresh_locks.clear()
    yield
    get_settings.cache_clear()
    coros_oauth._not_connected_cache.clear()
    coros_oauth._refresh_locks.clear()


@pytest.mark.asyncio
async def test_get_or_register_client_reuses_existing_row(monkeypatch):
    session = ScriptedSession(results=[{"client_id": "existing-id", "client_secret": None}])
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    client_id, client_secret = await coros_oauth._get_or_register_client()

    assert client_id == "existing-id"
    assert client_secret is None
    assert _writes(session) == []  # no registration call needed


@pytest.mark.asyncio
async def test_get_or_register_client_registers_when_absent(monkeypatch):
    session = ScriptedSession(results=[None])
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
    session = ScriptedSession(results=[None])  # delete...returning finds nothing
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    with pytest.raises(coros_oauth.CorosOAuthError, match="Unknown or already-used"):
        await coros_oauth.handle_callback(code="abc", state="does-not-exist")


@pytest.mark.asyncio
async def test_handle_callback_rejects_expired_state(monkeypatch):
    session = ScriptedSession(
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
    session = ScriptedSession(results=[None])
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    status = await coros_oauth.get_status("user-123")

    assert status.connected is False


@pytest.mark.asyncio
async def test_get_access_token_returns_none_when_not_connected(monkeypatch):
    session = ScriptedSession(results=[None])
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    token = await coros_oauth.get_access_token("user-123")

    assert token is None


@pytest.mark.asyncio
async def test_disconnect_deletes_local_row_even_if_revocation_fails(monkeypatch):
    session = ScriptedSession(
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
async def test_disconnect_drops_pooled_coros_chat_session(monkeypatch):
    session = ScriptedSession(results=[None])  # no stored row -> revocation call is skipped
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)
    discard = AsyncMock()
    monkeypatch.setattr(coros_oauth.coros_mcp, "discard", discard)

    await coros_oauth.disconnect("user-123")

    discard.assert_awaited_once_with("user-123")


@pytest.mark.asyncio
async def test_disconnect_raises_coros_oauth_error_on_db_failure(monkeypatch):
    class _FailingCommitSession(ScriptedSession):
        async def commit(self):
            raise SQLAlchemyError("connection lost")

    session = _FailingCommitSession(results=[None])  # no active integration -> revocation call is skipped
    monkeypatch.setattr(coros_oauth, "get_session", lambda: session)

    with pytest.raises(coros_oauth.CorosOAuthError, match="Failed to delete local COROS integration"):
        await coros_oauth.disconnect("user-123")


@pytest.mark.asyncio
async def test_is_connected_reflects_stored_row(monkeypatch):
    monkeypatch.setattr(
        coros_oauth, "get_session", lambda: ScriptedSession(results=[{"connected_at": datetime.now(UTC)}])
    )
    assert await coros_oauth.is_connected("user-123") is True

    monkeypatch.setattr(coros_oauth, "get_session", lambda: ScriptedSession(results=[None]))
    assert await coros_oauth.is_connected("user-123") is False


class _TokenStore:
    """Shared fake user_integrations row; every get_session() reads the current token state."""

    def __init__(self, expires_at):
        self.row = {
            "access_token_encrypted": "old-access",
            "refresh_token_encrypted": "old-refresh",
            "expires_at": expires_at,
        }

    def session(self):
        store = self

        class _S(FakeSession):
            async def handle(self, stmt):
                return row_result(store.row)

        return _S()


def _install_token_store(monkeypatch, expires_at) -> tuple[_TokenStore, AsyncMock]:
    store = _TokenStore(expires_at)
    monkeypatch.setattr(coros_oauth, "get_session", store.session)
    monkeypatch.setattr(coros_oauth, "decrypt_token", lambda v: v)

    async def _refresh(user_id, refresh_token):
        await asyncio.sleep(0.01)  # let a concurrent caller reach the lock meanwhile
        store.row = {
            "access_token_encrypted": "new-access",
            "refresh_token_encrypted": "new-refresh",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        }
        return "new-access"

    refresh = AsyncMock(side_effect=_refresh)
    monkeypatch.setattr(coros_oauth, "_refresh", refresh)
    return store, refresh


@pytest.mark.asyncio
async def test_get_access_token_uses_current_token_outside_margin(monkeypatch):
    _, refresh = _install_token_store(
        monkeypatch, datetime.now(UTC) + coros_oauth.REFRESH_MARGIN + timedelta(minutes=1)
    )

    assert await coros_oauth.get_access_token("user-123") == "old-access"
    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_access_token_refreshes_within_margin_before_expiry(monkeypatch):
    _, refresh = _install_token_store(monkeypatch, datetime.now(UTC) + timedelta(minutes=2))

    assert await coros_oauth.get_access_token("user-123") == "new-access"
    refresh.assert_awaited_once_with("user-123", "old-refresh")


@pytest.mark.asyncio
async def test_get_access_token_refreshes_after_expiry(monkeypatch):
    _, refresh = _install_token_store(monkeypatch, datetime.now(UTC) - timedelta(minutes=1))

    assert await coros_oauth.get_access_token("user-123") == "new-access"
    refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_get_access_token_refreshes_once(monkeypatch):
    _, refresh = _install_token_store(monkeypatch, datetime.now(UTC) + timedelta(minutes=1))

    tokens = await asyncio.gather(*(coros_oauth.get_access_token("user-123") for _ in range(3)))

    assert tokens == ["new-access"] * 3
    refresh.assert_awaited_once_with("user-123", "old-refresh")


@pytest.mark.asyncio
async def test_failed_early_refresh_falls_back_to_unexpired_token(monkeypatch):
    _, refresh = _install_token_store(monkeypatch, datetime.now(UTC) + timedelta(minutes=2))
    refresh.side_effect = coros_oauth.CorosOAuthError("COROS down")

    assert await coros_oauth.get_access_token("user-123") == "old-access"


@pytest.mark.asyncio
async def test_failed_refresh_of_expired_token_raises(monkeypatch):
    _, refresh = _install_token_store(monkeypatch, datetime.now(UTC) - timedelta(minutes=1))
    refresh.side_effect = coros_oauth.CorosOAuthError("COROS down")

    with pytest.raises(coros_oauth.CorosOAuthError):
        await coros_oauth.get_access_token("user-123")
