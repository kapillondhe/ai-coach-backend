import logging

from fastapi import APIRouter, Header, HTTPException

from app.core.config import get_settings
from app.services import coros_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/sync", tags=["internal"])


@router.post("/run")
async def run_sync(x_internal_sync_secret: str | None = Header(default=None)) -> dict[str, int]:
    settings = get_settings()
    if not settings.internal_sync_secret:
        raise HTTPException(status_code=503, detail="Internal sync is not configured")
    if x_internal_sync_secret != settings.internal_sync_secret:
        raise HTTPException(status_code=401, detail="Invalid sync secret")

    return await coros_sync.run_periodic_sync()
