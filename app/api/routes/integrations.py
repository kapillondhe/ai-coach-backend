import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.core.auth import require_user_id
from app.core.config import get_settings
from app.services import coros_oauth, coros_sync
from app.services.coros_oauth import CorosOAuthError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrations", tags=["integrations"])

# Client-facing 502 details: CorosOAuthError messages can carry COROS's raw response
# body, so they go to the server log (logger.exception) only.
_CONNECT_FAILED_DETAIL = "Couldn't start the COROS connection. Please try again."
_DISCONNECT_FAILED_DETAIL = "Couldn't disconnect COROS. Please try again."


class CorosAuthorizationResponse(BaseModel):
    authorization_url: str


class CorosStatusResponse(BaseModel):
    connected: bool
    connected_at: datetime | None
    syncing: bool
    last_synced_at: datetime | None


class CorosDisconnectResponse(BaseModel):
    status: str


@router.post("/coros/connect")
async def coros_connect(user_id: str = Depends(require_user_id)) -> CorosAuthorizationResponse:
    try:
        url = await coros_oauth.build_authorization_url(user_id)
    except CorosOAuthError as exc:
        logger.exception("Failed to start COROS connect flow")
        raise HTTPException(status_code=502, detail=_CONNECT_FAILED_DETAIL) from exc
    return CorosAuthorizationResponse(authorization_url=url)


@router.get("/coros/callback")
async def coros_callback(code: str, state: str, background_tasks: BackgroundTasks) -> RedirectResponse:
    settings = get_settings()
    try:
        user_id = await coros_oauth.handle_callback(code, state)
    except CorosOAuthError:
        logger.exception("COROS OAuth callback failed")
        return RedirectResponse(url=f"{settings.frontend_url}/profile?coros=error")
    background_tasks.add_task(coros_sync.backfill_user, user_id)
    return RedirectResponse(url=f"{settings.frontend_url}/profile?coros=connected")


@router.get("/coros/status")
async def coros_status(user_id: str = Depends(require_user_id)) -> CorosStatusResponse:
    status = await coros_oauth.get_status(user_id)
    sync_status = await coros_sync.get_sync_status(user_id) if status.connected else None
    return CorosStatusResponse(
        connected=status.connected,
        connected_at=status.connected_at,
        syncing=sync_status.last_backfill_completed_at is None if sync_status else status.connected,
        last_synced_at=sync_status.last_synced_at if sync_status else None,
    )


@router.post("/coros/disconnect")
async def coros_disconnect(user_id: str = Depends(require_user_id)) -> CorosDisconnectResponse:
    try:
        await coros_oauth.disconnect(user_id)
    except CorosOAuthError as exc:
        logger.exception("Failed to disconnect COROS integration")
        raise HTTPException(status_code=502, detail=_DISCONNECT_FAILED_DETAIL) from exc
    return CorosDisconnectResponse(status="disconnected")
