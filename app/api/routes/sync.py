import logging

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.core.config import get_settings
from app.services import coros_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/sync", tags=["internal"])


class SyncRunResponse(BaseModel):
    synced: int
    failed: int


@router.post("/run")
async def run_sync(x_internal_sync_secret: str | None = Header(default=None)) -> SyncRunResponse:
    settings = get_settings()
    if not settings.internal_sync_secret:
        raise HTTPException(status_code=503, detail="Internal sync is not configured")
    if x_internal_sync_secret != settings.internal_sync_secret:
        raise HTTPException(status_code=401, detail="Invalid sync secret")

    result = await coros_sync.run_periodic_sync()
    return SyncRunResponse(synced=result["synced"], failed=result["failed"])
