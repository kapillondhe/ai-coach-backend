from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.db import get_session
from app.core.models import Profile

EDITABLE_FIELDS = (
    "name",
    "weight_kg",
    "injury_notes",
)

_RETURNED_COLUMNS = (Profile.name, Profile.weight_kg, Profile.injury_notes, Profile.field_sources)


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


def _get_or_create_stmt(user_id: str):
    """Insert an empty profile, or return the existing one, in a single statement.

    The no-op DO UPDATE (rather than DO NOTHING) makes RETURNING yield the row on
    conflict too, and locks it for the rest of the transaction.
    """
    now = datetime.now(UTC)
    stmt = pg_insert(Profile).values(user_id=user_id, field_sources={}, created_at=now, updated_at=now)
    stmt = stmt.on_conflict_do_update(index_elements=[Profile.user_id], set_={"user_id": stmt.excluded.user_id})
    return stmt.returning(*_RETURNED_COLUMNS)


async def get_or_create_profile(user_id: str) -> ProfileData:
    async with get_session() as session:
        row = (await session.execute(_get_or_create_stmt(user_id))).one()
        await session.commit()
    return _row_to_data(user_id, row)


def _validate_updates(updates: dict[str, object]) -> None:
    unknown = set(updates) - set(EDITABLE_FIELDS)
    if unknown:
        raise ProfileError(f"Unknown profile field(s): {sorted(unknown)}")


async def update_profile(user_id: str, updates: dict[str, object], *, source: str = "user") -> ProfileData:
    """Apply `updates` and record `source` as each edited field's source, in two statements.

    Both run in one transaction: the get-or-create upsert locks the row, so a concurrent
    update can't slip in between reading field_sources and writing the merged value.
    """
    _validate_updates(updates)
    async with get_session() as session:
        existing = (await session.execute(_get_or_create_stmt(user_id))).one()
        sources = {**(existing.field_sources or {}), **dict.fromkeys(updates, source)}
        row = (
            await session.execute(
                update(Profile)
                .where(Profile.user_id == user_id)
                .values(**updates, field_sources=sources, updated_at=datetime.now(UTC))
                .returning(*_RETURNED_COLUMNS)
            )
        ).one()
        await session.commit()
    return _row_to_data(user_id, row)
