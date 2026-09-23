
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.db import get_session
from app.core.models import Profile

EDITABLE_FIELDS = (
    "name",
    "weight_kg",
    "injury_notes",
)


class ProfileError(ValueError):
    pass


@dataclass
class ProfileData:
    user_id: str
    name: str | None = None
    weight_kg: float | None = None
    injury_notes: str | None = None
    field_sources: dict[str, str] = field(default_factory=dict)


def _row_to_data(user_id: str, row) -> ProfileData:
    return ProfileData(
        user_id=user_id,
        name=row.name,
        weight_kg=row.weight_kg,
        injury_notes=row.injury_notes,
        field_sources=dict(row.field_sources or {}),
    )


async def _select_profile(user_id: str):
    async with get_session() as session:
        return (
            await session.execute(
                select(
                    Profile.name,
                    Profile.weight_kg,
                    Profile.injury_notes,
                    Profile.field_sources,
                ).where(Profile.user_id == user_id)
            )
        ).first()


async def get_or_create_profile(user_id: str) -> ProfileData:
    now = datetime.now(UTC)
    stmt = pg_insert(Profile).values(
        user_id=user_id, field_sources={}, created_at=now, updated_at=now
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=[Profile.user_id])
    async with get_session() as session:
        await session.execute(stmt)
        await session.commit()

    row = await _select_profile(user_id)
    assert row is not None
    return _row_to_data(user_id, row)


def _validate_updates(updates: dict[str, object]) -> None:
    unknown = set(updates) - set(EDITABLE_FIELDS)
    if unknown:
        raise ProfileError(f"Unknown profile field(s): {sorted(unknown)}")


async def update_profile(user_id: str, updates: dict[str, object], *, source: str = "user") -> ProfileData:
    _validate_updates(updates)
    await get_or_create_profile(user_id)  # ensure a row exists

    existing_row = await _select_profile(user_id)
    assert existing_row is not None
    sources = dict(existing_row.field_sources or {})
    for key in updates:
        if key in EDITABLE_FIELDS:
            sources[key] = source

    values = dict(updates)
    values["field_sources"] = sources
    values["updated_at"] = datetime.now(UTC)

    async with get_session() as session:
        await session.execute(update(Profile).where(Profile.user_id == user_id).values(**values))
        await session.commit()

    row = await _select_profile(user_id)
    assert row is not None
    return _row_to_data(user_id, row)
