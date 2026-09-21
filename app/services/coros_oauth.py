import base64
import contextlib
import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.core.db import get_session
from app.core.models import OAuthClient, OAuthState, UserIntegration
from app.services.token_crypto import decrypt_token, encrypt_token

PROVIDER = "coros"
SCOPES = "openid mcp.tools offline_access"
STATE_TTL = timedelta(minutes=10)

ISSUER = "https://mcpus.coros.com"
AUTHORIZATION_ENDPOINT = f"{ISSUER}/oauth2/authorize"
TOKEN_ENDPOINT = f"{ISSUER}/oauth2/token"
REVOCATION_ENDPOINT = f"{ISSUER}/oauth2/revoke"
REGISTRATION_ENDPOINT = f"{ISSUER}/connect/register"


class CorosOAuthError(Exception):
    pass


@dataclass
class ConnectionStatus:
    connected: bool
    connected_at: datetime | None = None


def _generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) per RFC 7636, S256 method."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


async def _get_or_register_client() -> tuple[str, str | None]:
    async with get_session() as session:
        row = (
            await session.execute(
                select(OAuthClient.client_id, OAuthClient.client_secret).where(OAuthClient.provider == PROVIDER)
            )
        ).first()
    if row:
        return row.client_id, row.client_secret

    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                REGISTRATION_ENDPOINT,
                json={
                    "redirect_uris": [settings.coros_redirect_uri],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "client_name": "AI Coach",
                },
            )
    except httpx.HTTPError as exc:
        raise CorosOAuthError(f"COROS client registration request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise CorosOAuthError(f"COROS client registration failed: {resp.status_code} {resp.text}")

    data = resp.json()
    client_id = data["client_id"]
    client_secret = data.get("client_secret")

    stmt = pg_insert(OAuthClient).values(
        provider=PROVIDER, client_id=client_id, client_secret=client_secret, created_at=datetime.now(UTC)
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[OAuthClient.provider],
        set_={"client_id": stmt.excluded.client_id, "client_secret": stmt.excluded.client_secret},
    )
    async with get_session() as session:
        await session.execute(stmt)
        await session.commit()
    return client_id, client_secret


async def build_authorization_url(user_id: str) -> str:
    settings = get_settings()
    client_id, _ = await _get_or_register_client()
    code_verifier, code_challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(32)

    now = datetime.now(UTC)
    async with get_session() as session:
        session.add(
            OAuthState(
                state=state,
                provider=PROVIDER,
                user_id=user_id,
                code_verifier=code_verifier,
                created_at=now,
                expires_at=now + STATE_TTL,
            )
        )
        await session.commit()

    params = httpx.QueryParams(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": settings.coros_redirect_uri,
            "scope": SCOPES,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{AUTHORIZATION_ENDPOINT}?{params}"


async def handle_callback(code: str, state: str) -> str:
    async with get_session() as session:
        state_row = (
            await session.execute(
                delete(OAuthState)
                .where(OAuthState.state == state, OAuthState.provider == PROVIDER)
                .returning(OAuthState.user_id, OAuthState.code_verifier, OAuthState.expires_at)
            )
        ).first()
        await session.commit()
    if state_row is None:
        raise CorosOAuthError("Unknown or already-used OAuth state")
    if state_row.expires_at < datetime.now(UTC):
        raise CorosOAuthError("OAuth state expired")

    user_id = str(state_row.user_id)
    code_verifier = state_row.code_verifier

    settings = get_settings()
    client_id, client_secret = await _get_or_register_client()

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.coros_redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(TOKEN_ENDPOINT, data=data)
    except httpx.HTTPError as exc:
        raise CorosOAuthError(f"COROS token exchange request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise CorosOAuthError(f"COROS token exchange failed: {resp.status_code} {resp.text}")

    token_data = resp.json()
    access_token = token_data["access_token"]
    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get("expires_in")
    expires_at = datetime.now(UTC) + timedelta(seconds=expires_in) if expires_in else None
    now = datetime.now(UTC)

    stmt = pg_insert(UserIntegration).values(
        user_id=user_id,
        provider=PROVIDER,
        access_token_encrypted=encrypt_token(access_token),
        refresh_token_encrypted=encrypt_token(refresh_token) if refresh_token else None,
        scope=token_data.get("scope"),
        expires_at=expires_at,
        connected_at=now,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[UserIntegration.user_id, UserIntegration.provider],
        set_={
            "access_token_encrypted": stmt.excluded.access_token_encrypted,
            "refresh_token_encrypted": stmt.excluded.refresh_token_encrypted,
            "scope": stmt.excluded.scope,
            "expires_at": stmt.excluded.expires_at,
            "updated_at": stmt.excluded.updated_at,
        },
    )
    async with get_session() as session:
        await session.execute(stmt)
        await session.commit()
    return user_id


async def get_status(user_id: str) -> ConnectionStatus:
    async with get_session() as session:
        row = (
            await session.execute(
                select(UserIntegration.connected_at).where(
                    UserIntegration.user_id == user_id, UserIntegration.provider == PROVIDER
                )
            )
        ).first()
    if row is None:
        return ConnectionStatus(connected=False)
    return ConnectionStatus(connected=True, connected_at=row.connected_at)


async def get_access_token(user_id: str) -> str | None:
    """Return a usable (refreshing if needed) access token for this user's COROS connection, or None if not connected."""
    async with get_session() as session:
        row = (
            await session.execute(
                select(
                    UserIntegration.access_token_encrypted,
                    UserIntegration.refresh_token_encrypted,
                    UserIntegration.expires_at,
                ).where(UserIntegration.user_id == user_id, UserIntegration.provider == PROVIDER)
            )
        ).first()
    if row is None:
        return None

    if row.expires_at is not None and row.expires_at < datetime.now(UTC) and row.refresh_token_encrypted:
        return await _refresh(user_id, decrypt_token(row.refresh_token_encrypted))

    return decrypt_token(row.access_token_encrypted)


async def _refresh(user_id: str, refresh_token: str) -> str:
    client_id, client_secret = await _get_or_register_client()
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if client_secret:
        data["client_secret"] = client_secret

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(TOKEN_ENDPOINT, data=data)
    except httpx.HTTPError as exc:
        raise CorosOAuthError(f"COROS token refresh request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise CorosOAuthError(f"COROS token refresh failed: {resp.status_code} {resp.text}")

    token_data = resp.json()
    access_token = token_data["access_token"]
    new_refresh_token = token_data.get("refresh_token", refresh_token)
    expires_in = token_data.get("expires_in")
    expires_at = datetime.now(UTC) + timedelta(seconds=expires_in) if expires_in else None

    async with get_session() as session:
        await session.execute(
            update(UserIntegration)
            .where(UserIntegration.user_id == user_id, UserIntegration.provider == PROVIDER)
            .values(
                access_token_encrypted=encrypt_token(access_token),
                refresh_token_encrypted=encrypt_token(new_refresh_token),
                expires_at=expires_at,
                updated_at=datetime.now(UTC),
            )
        )
        await session.commit()
    return access_token


async def disconnect(user_id: str) -> None:
    """Revoke the stored token at COROS (best-effort) and delete it locally."""
    async with get_session() as session:
        row = (
            await session.execute(
                select(UserIntegration.access_token_encrypted).where(
                    UserIntegration.user_id == user_id, UserIntegration.provider == PROVIDER
                )
            )
        ).first()

    if row is not None:
        client_id, client_secret = await _get_or_register_client()
        data = {"token": decrypt_token(row.access_token_encrypted), "client_id": client_id}
        if client_secret:
            data["client_secret"] = client_secret
        async with httpx.AsyncClient(timeout=10) as client:
            # Best-effort: don't block disconnect on COROS's revocation endpoint being
            # slow/unavailable — the local row is deleted regardless, below.
            with contextlib.suppress(httpx.HTTPError):
                await client.post(REVOCATION_ENDPOINT, data=data)

    try:
        async with get_session() as session:
            await session.execute(
                delete(UserIntegration).where(
                    UserIntegration.user_id == user_id, UserIntegration.provider == PROVIDER
                )
            )
            await session.commit()
    except SQLAlchemyError as exc:
        raise CorosOAuthError(f"Failed to delete local COROS integration: {exc}") from exc
