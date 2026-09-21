from unittest.mock import MagicMock

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core import auth as auth_module
from app.core.auth import get_current_user_id, require_user_id


@pytest.fixture(autouse=True)
def _clear_caches():
    from app.core.config import get_settings

    get_settings.cache_clear()
    auth_module._jwks_client.cache_clear()
    yield
    get_settings.cache_clear()
    auth_module._jwks_client.cache_clear()


def _request(headers: dict[str, str] | None = None):
    """Minimal fastapi.Request-like object with headers, for get_current_user_id."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    return Request(scope)


def test_anonymous_when_supabase_url_unset(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8100/mcp")
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    assert get_current_user_id(_request({"authorization": "Bearer whatever"})) is None


def test_anonymous_when_no_authorization_header(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8100/mcp")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")

    assert get_current_user_id(_request()) is None


def test_anonymous_when_header_not_bearer(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8100/mcp")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")

    assert get_current_user_id(_request({"authorization": "Basic abc"})) is None


def test_anonymous_on_invalid_token(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8100/mcp")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setattr(
        auth_module,
        "_jwks_client",
        lambda supabase_url: (_ for _ in ()).throw(jwt.PyJWKClientError("boom")),
    )

    assert get_current_user_id(_request({"authorization": "Bearer not-a-real-token"})) is None


def test_resolves_user_id_from_valid_token(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8100/mcp")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")

    fake_signing_key = MagicMock(key="fake-key")
    fake_jwks_client = MagicMock()
    fake_jwks_client.get_signing_key_from_jwt.return_value = fake_signing_key
    monkeypatch.setattr(auth_module, "_jwks_client", lambda supabase_url: fake_jwks_client)
    monkeypatch.setattr(
        auth_module.jwt,
        "decode",
        lambda token, key, algorithms, audience: {"sub": "user-abc", "aud": audience},
    )

    assert get_current_user_id(_request({"authorization": "Bearer a.valid.jwt"})) == "user-abc"


def test_require_user_id_raises_401_when_anonymous():
    with pytest.raises(HTTPException) as exc_info:
        require_user_id(user_id=None)
    assert exc_info.value.status_code == 401


def test_require_user_id_passes_through_when_signed_in():
    assert require_user_id(user_id="user-123") == "user-123"


def test_chat_route_works_anonymously_without_authorization_header():
    # /coach/chat must never 401 purely for lacking a token — anonymous is
    # a first-class, permanent mode (docs/user-identity-design.md).
    from app.main import app

    client = TestClient(app)
    response = client.post("/api/coach/chat", json={"message": "hello"})

    assert response.status_code != 401
