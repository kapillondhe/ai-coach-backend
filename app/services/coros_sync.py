import logging
import uuid
from collections.abc import Callable, Hashable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from json import JSONDecodeError
from json import loads as json_loads

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import get_settings
from app.core.db import get_session
from app.core.models import (
    IntegrationSyncState,
    SyncedActivity,
    SyncedDailyMetric,
    SyncedSnapshot,
    SyncStatusValue,
    UserIntegration,
)
from app.services import coros_oauth
from app.services import coros_sync_parsers as parsers
from app.services.coros_oauth import PROVIDER

logger = logging.getLogger(__name__)

BACKFILL_WINDOW_DAYS = 28
PERIODIC_SYNC_WINDOW_DAYS = 2  # cheap incremental catch-up, not a full re-pull each run
# Rows per multi-row upsert statement; keeps bind params well under asyncpg's 32767 cap.
UPSERT_BATCH_SIZE = 500


class CorosSyncError(Exception):
    pass


@asynccontextmanager
async def _mcp_session(access_token: str):
    settings = get_settings()
    http_client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {access_token}"}, timeout=30)
    async with (
        streamable_http_client(settings.coros_mcp_server_url, http_client=http_client) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def _call_tool_text(session: ClientSession, name: str, arguments: dict) -> str:
    result = await session.call_tool(name, arguments)
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        raise CorosSyncError(f"COROS MCP tool {name} returned an error: {result.content}")
    if not result.content:
        return ""
    first = result.content[0]
    text = getattr(first, "text", None)
    if text is None:
        raise CorosSyncError(f"COROS MCP tool {name} returned non-text content")
    # Tool responses are JSON-encoded strings (a quoted string containing the
    # narrative text) — decode once, falling back to the raw text if it isn't.
    try:
        decoded = json_loads(text)
        return decoded if isinstance(decoded, str) else text
    except (JSONDecodeError, TypeError):
        return text


async def _set_sync_status(user_id: str, *, status: SyncStatusValue, error: str | None = None) -> None:
    stmt = pg_insert(IntegrationSyncState).values(
        user_id=user_id,
        provider=PROVIDER,
        status=status,
        last_error=error if status == SyncStatusValue.ERROR else None,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[IntegrationSyncState.user_id, IntegrationSyncState.provider],
        set_={"status": stmt.excluded.status, "last_error": stmt.excluded.last_error},
    )
    async with get_session() as db:
        await db.execute(stmt)
        await db.commit()


async def _mark_synced(user_id: str, *, backfill: bool) -> None:
    now = datetime.now(UTC)
    values = {
        "status": SyncStatusValue.IDLE,
        "last_synced_at": now,
        "last_error": None,
    }
    if backfill:
        values["last_backfill_completed_at"] = now
    stmt = pg_insert(IntegrationSyncState).values(user_id=user_id, provider=PROVIDER, **values)
    update_set = dict(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[IntegrationSyncState.user_id, IntegrationSyncState.provider],
        set_={k: getattr(stmt.excluded, k) for k in update_set},
    )
    async with get_session() as db:
        await db.execute(stmt)
        await db.commit()


def _batches[T](rows: list[T]) -> Iterator[list[T]]:
    for i in range(0, len(rows), UPSERT_BATCH_SIZE):
        yield rows[i : i + UPSERT_BATCH_SIZE]


def _last_per_key[T](rows: list[T], key: Callable[[T], Hashable]) -> list[T]:
    """Drop earlier duplicates of a conflict key, keeping the last.

    Postgres rejects one multi-row ON CONFLICT DO UPDATE that touches the same row
    twice; the old one-statement-per-row loop let the last duplicate win.
    """
    return list({key(r): r for r in rows}.values())


def _epoch_to_utc(epoch: int | None) -> datetime | None:
    return datetime.fromtimestamp(epoch, tz=UTC) if epoch else None


async def _upsert_activities(user_id: str, activities: list[parsers.ParsedActivity]) -> int:
    if not activities:
        return 0
    now = datetime.now(UTC)
    rows = [
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "provider": PROVIDER,
            "external_id": act.external_id,
            "discipline": act.discipline,
            "sport_type_code": act.sport_type_code,
            "started_at": _epoch_to_utc(act.started_at_epoch),
            "ended_at": _epoch_to_utc(act.ended_at_epoch),
            "duration_seconds": act.duration_seconds,
            "distance_km": act.distance_km,
            "avg_pace_sec_per_km": act.avg_pace_sec_per_km,
            "avg_hr": act.avg_hr,
            "calories": act.calories,
            "raw_payload": {"text": act.raw_text},
            "synced_at": now,
        }
        for act in _last_per_key(activities, lambda a: a.external_id)
    ]
    async with get_session() as db:
        for batch in _batches(rows):
            stmt = pg_insert(SyncedActivity).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=[SyncedActivity.user_id, SyncedActivity.provider, SyncedActivity.external_id],
                set_={
                    "discipline": stmt.excluded.discipline,
                    "sport_type_code": stmt.excluded.sport_type_code,
                    "started_at": stmt.excluded.started_at,
                    "ended_at": stmt.excluded.ended_at,
                    "duration_seconds": stmt.excluded.duration_seconds,
                    "distance_km": stmt.excluded.distance_km,
                    "avg_pace_sec_per_km": stmt.excluded.avg_pace_sec_per_km,
                    "avg_hr": stmt.excluded.avg_hr,
                    "calories": stmt.excluded.calories,
                    "raw_payload": stmt.excluded.raw_payload,
                    "synced_at": stmt.excluded.synced_at,
                },
            )
            await db.execute(stmt)
        await db.commit()
    return len(rows)


async def _upsert_daily_metrics(user_id: str, metric_type: str, values: list[parsers.ParsedDailyValue]) -> int:
    if not values:
        return 0
    now = datetime.now(UTC)
    rows = [
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "provider": PROVIDER,
            "metric_type": metric_type,
            "date": v.date,
            "value": v.value,
            "extra": v.extra,
            "synced_at": now,
        }
        for v in _last_per_key(values, lambda v: v.date)
    ]
    async with get_session() as db:
        for batch in _batches(rows):
            stmt = pg_insert(SyncedDailyMetric).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    SyncedDailyMetric.user_id,
                    SyncedDailyMetric.provider,
                    SyncedDailyMetric.metric_type,
                    SyncedDailyMetric.date,
                ],
                set_={"value": stmt.excluded.value, "extra": stmt.excluded.extra, "synced_at": stmt.excluded.synced_at},
            )
            await db.execute(stmt)
        await db.commit()
    return len(rows)


async def _insert_snapshot(user_id: str, snapshot_type: str, data: dict) -> None:
    now = datetime.now(UTC)
    async with get_session() as db:
        db.add(
            SyncedSnapshot(
                id=str(uuid.uuid4()),
                user_id=user_id,
                provider=PROVIDER,
                snapshot_type=snapshot_type,
                data=data,
                captured_at=now,
            )
        )
        await db.commit()


async def sync_user(user_id: str, *, days: int, backfill: bool = False) -> None:
    """Pull and store `days` worth of COROS data for one user. Raises CorosSyncError on failure.

    `backfill=True` marks this as the initial full pull (sets last_backfill_completed_at).
    """
    access_token = await coros_oauth.get_access_token(user_id)
    if access_token is None:
        raise CorosSyncError(f"User {user_id} has no active COROS connection")

    await _set_sync_status(user_id, status=SyncStatusValue.SYNCING)

    # UTC, like every other "today" in the backend (the chat prompt's date included).
    today = datetime.now(UTC).date()
    start = today - timedelta(days=days)

    try:
        async with _mcp_session(access_token) as session:
            sport_text = await _call_tool_text(
                session,
                "querySportRecords",
                {
                    "startDate": start.strftime("%Y%m%d"),
                    "endDate": today.strftime("%Y%m%d"),
                    "sportTypeCodes": None,
                    "minDistanceKm": None,
                    "maxDistanceKm": None,
                    "minDurationMinutes": None,
                    "maxDurationMinutes": None,
                    "maxAveragePace": None,
                    "locationKeyword": None,
                    "limit": 200,
                },
            )
            resting_hr_text = await _call_tool_text(session, "queryRestingHeartRate", {"days": days})
            avg_hr_text = await _call_tool_text(session, "queryAvgHeartRate", {"days": days})
            load_text = await _call_tool_text(session, "queryTrainingLoadAssessment", {"days": days})
            recovery_text = await _call_tool_text(session, "queryRecoveryStatus", {})
            fitness_text = await _call_tool_text(session, "queryFitnessAssessmentOverview", {})
    except CorosSyncError:
        await _set_sync_status(user_id, status=SyncStatusValue.ERROR, error="COROS MCP tool call failed")
        raise
    except Exception as exc:
        await _set_sync_status(user_id, status=SyncStatusValue.ERROR, error=str(exc))
        raise CorosSyncError(f"COROS MCP session failed for user {user_id}: {exc}") from exc

    try:
        activities = parsers.parse_sport_records(sport_text)
        resting_hr = parsers.parse_resting_heart_rate(resting_hr_text)
        avg_hr = parsers.parse_avg_heart_rate(avg_hr_text)
        load = parsers.parse_training_load_assessment(load_text)
        recovery = parsers.parse_recovery_status(recovery_text)
        fitness = parsers.parse_fitness_assessment_overview(fitness_text)

        await _upsert_activities(user_id, activities)
        await _upsert_daily_metrics(user_id, "resting_hr", resting_hr)
        await _upsert_daily_metrics(user_id, "avg_hr", avg_hr)
        await _upsert_daily_metrics(user_id, "training_load", load)
        if recovery:
            await _insert_snapshot(user_id, "recovery_status", recovery)
        if fitness:
            await _insert_snapshot(user_id, "fitness_assessment", fitness)
    except Exception as exc:
        await _set_sync_status(user_id, status=SyncStatusValue.ERROR, error=f"Parsing/storage failed: {exc}")
        raise CorosSyncError(f"Failed to parse/store COROS data for user {user_id}: {exc}") from exc

    await _mark_synced(user_id, backfill=backfill)


async def backfill_user(user_id: str) -> None:
    """Initial sync on COROS connect — trailing BACKFILL_WINDOW_DAYS. Never raises: logs and marks error."""
    try:
        await sync_user(user_id, days=BACKFILL_WINDOW_DAYS, backfill=True)
    except Exception:
        # Runs as a fire-and-forget BackgroundTask: a DB/token error (not just
        # CorosSyncError) must not escape into the request's background runner.
        logger.exception("COROS backfill failed for user %s", user_id)


async def run_periodic_sync() -> dict[str, int]:
    """Re-sync every connected user's recent window. One user's failure never stops the batch."""
    async with get_session() as db:
        user_ids = (
            (await db.execute(select(UserIntegration.user_id).where(UserIntegration.provider == PROVIDER)))
            .scalars()
            .all()
        )

    succeeded, failed = 0, 0
    for user_id in user_ids:
        try:
            await sync_user(user_id, days=PERIODIC_SYNC_WINDOW_DAYS)
            succeeded += 1
        except CorosSyncError:
            logger.exception("Periodic COROS sync failed for user %s", user_id)
            failed += 1
    return {"synced": succeeded, "failed": failed}


@dataclass
class SyncStatus:
    status: SyncStatusValue
    last_synced_at: datetime | None
    last_backfill_completed_at: datetime | None
    last_error: str | None


async def get_sync_status(user_id: str) -> SyncStatus | None:
    async with get_session() as db:
        row = (
            await db.execute(
                select(
                    IntegrationSyncState.status,
                    IntegrationSyncState.last_synced_at,
                    IntegrationSyncState.last_backfill_completed_at,
                    IntegrationSyncState.last_error,
                ).where(IntegrationSyncState.user_id == user_id, IntegrationSyncState.provider == PROVIDER)
            )
        ).first()
    if row is None:
        return None
    return SyncStatus(
        status=SyncStatusValue(row.status),
        last_synced_at=row.last_synced_at,
        last_backfill_completed_at=row.last_backfill_completed_at,
        last_error=row.last_error,
    )
