import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import require_user_id
from app.services import profile as profile_service
from app.services.profile import ProfileError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile", tags=["profile"])


class ProfileResponse(BaseModel):
    name: str | None
    weight_kg: float | None
    injury_notes: str | None
    field_sources: dict[str, str]


class ProfileUpdateRequest(BaseModel):
    name: str | None = Field(default=None)
    weight_kg: float | None = Field(default=None)
    injury_notes: str | None = Field(default=None)


def _to_response(data: profile_service.ProfileData) -> ProfileResponse:
    return ProfileResponse(
        name=data.name,
        weight_kg=data.weight_kg,
        injury_notes=data.injury_notes,
        field_sources=data.field_sources,
    )


@router.get("")
async def get_profile(user_id: str = Depends(require_user_id)) -> ProfileResponse:
    data = await profile_service.get_or_create_profile(user_id)
    return _to_response(data)


@router.patch("")
async def patch_profile(
    body: ProfileUpdateRequest, user_id: str = Depends(require_user_id)
) -> ProfileResponse:
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        data = await profile_service.get_or_create_profile(user_id)
        return _to_response(data)
    try:
        data = await profile_service.update_profile(user_id, updates, source="user")
    except ProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(data)
