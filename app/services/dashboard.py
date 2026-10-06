import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import Row, select

from app.core.db import get_session
from app.core.models import IntegrationSyncState, SyncedDailyMetric, SyncedSnapshot, UserIntegration

TRAILING_WEEKS = 4


@dataclass
class SparklinePoint:
    date: str
    value: float | None


@dataclass
class TrendCard:
    id: str
    label: str
    sparkline: list[SparklinePoint]
    takeaway: str
    latest_value: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class DashboardSummary:
    connected: bool
    syncing: bool = False
    trends: list[TrendCard] = field(default_factory=list)


async def _has_any_integration(user_id: str) -> bool:
    """Whether the user has connected ANY provider.

    Deliberately provider-generic: the dashboard reads the provider-keyed synced
    tables, so a future provider (e.g. Strava) shows up here without code changes.
    Use `coros_oauth.is_connected` when you need COROS specifically.
    """
    async with get_session() as db:
        row = (
            await db.execute(select(UserIntegration.provider).where(UserIntegration.user_id == user_id).limit(1))
        ).first()
    return row is not None


async def _get_any_sync_state(user_id: str) -> Row | None:
    """Most-recently-synced connected provider's sync state, if any."""
    async with get_session() as db:
        row = (
            await db.execute(
                select(
                    IntegrationSyncState.status,
                    IntegrationSyncState.last_synced_at,
                    IntegrationSyncState.last_backfill_completed_at,
                    IntegrationSyncState.last_error,
                )
                .where(IntegrationSyncState.user_id == user_id)
                .order_by(IntegrationSyncState.last_synced_at.desc().nullslast())
                .limit(1)
            )
        ).first()
    return row


async def _daily_metric_series(user_id: str, metric_type: str, since: date) -> list[tuple[str, float | None, dict]]:
    async with get_session() as db:
        rows = (
            await db.execute(
                select(SyncedDailyMetric.date, SyncedDailyMetric.value, SyncedDailyMetric.extra)
                .where(
                    SyncedDailyMetric.user_id == user_id,
                    SyncedDailyMetric.metric_type == metric_type,
                    SyncedDailyMetric.date >= since.isoformat(),
                )
                .order_by(SyncedDailyMetric.date)
            )
        ).all()
    return [(r.date, r.value, dict(r.extra or {})) for r in rows]


async def _latest_snapshot(user_id: str, snapshot_type: str) -> dict | None:
    async with get_session() as db:
        row = (
            await db.execute(
                select(SyncedSnapshot.data)
                .where(SyncedSnapshot.user_id == user_id, SyncedSnapshot.snapshot_type == snapshot_type)
                .order_by(SyncedSnapshot.captured_at.desc())
                .limit(1)
            )
        ).first()
    return dict(row.data) if row else None


def _resting_hr_takeaway(series: list[tuple[str, float | None, dict]]) -> str:
    values = [v for _, v, _ in series if v is not None]
    if len(values) < 4:
        return "Not enough resting heart rate data yet to show a trend."
    midpoint = len(values) // 2
    earlier_avg = sum(values[:midpoint]) / midpoint
    later_avg = sum(values[midpoint:]) / (len(values) - midpoint)
    delta = later_avg - earlier_avg
    if abs(delta) < 1:
        return "Resting heart rate has been stable."
    direction = "up" if delta > 0 else "down"
    return f"Resting heart rate is trending {direction} slightly over the last few weeks."


def _training_load_takeaway(series: list[tuple[str, float | None, dict]]) -> str:
    if not series:
        return "Not enough training load data yet."
    _, ratio, extra = series[-1]
    comment = extra.get("comment")
    if comment and ratio is not None:
        return f"{comment} — current load ratio is {ratio:.2f}."
    if comment:
        return str(comment)
    return "Training load data available, no trend comment from your device yet."


def _fitness_takeaway(snapshot: dict | None) -> str:
    if not snapshot:
        return "Connect a device and complete a few activities to see a fitness assessment."
    vo2 = snapshot.get("vo2max")
    threshold = snapshot.get("threshold_pace")
    if vo2 and threshold:
        return f"VO2max is {vo2}, threshold pace is {threshold}. This updates each sync — not yet a trend."
    return "Fitness assessment available — this updates each sync, not yet a trend."


async def get_dashboard_summary(user_id: str) -> DashboardSummary:
    # Independent reads run concurrently; each helper opens its own session, so no
    # session is shared across the gathered coroutines.
    connected, sync_state = await asyncio.gather(_has_any_integration(user_id), _get_any_sync_state(user_id))
    if not connected:
        return DashboardSummary(connected=False)
    if sync_state is None or sync_state.last_backfill_completed_at is None:
        return DashboardSummary(connected=True, syncing=True)

    # UTC, like every other "today" in the backend (the chat prompt's date included).
    since = datetime.now(UTC).date() - timedelta(weeks=TRAILING_WEEKS)

    resting_hr_series, training_load_series, fitness_snapshot = await asyncio.gather(
        _daily_metric_series(user_id, "resting_hr", since),
        _daily_metric_series(user_id, "training_load", since),
        _latest_snapshot(user_id, "fitness_assessment"),
    )

    trends = [
        TrendCard(
            id="resting_hr",
            label="Resting heart rate",
            sparkline=[SparklinePoint(date=d, value=v) for d, v, _ in resting_hr_series],
            takeaway=_resting_hr_takeaway(resting_hr_series),
            latest_value=resting_hr_series[-1][1] if resting_hr_series else None,
        ),
        TrendCard(
            id="training_load",
            label="Training load",
            sparkline=[SparklinePoint(date=d, value=v) for d, v, _ in training_load_series],
            takeaway=_training_load_takeaway(training_load_series),
            latest_value=training_load_series[-1][1] if training_load_series else None,
            extra=training_load_series[-1][2] if training_load_series else {},
        ),
        TrendCard(
            id="fitness_assessment",
            label="Fitness snapshot",
            sparkline=[],  # no vendor-side history for this metric — see docs/tasks/10
            takeaway=_fitness_takeaway(fitness_snapshot),
            extra=fitness_snapshot or {},
        ),
    ]

    return DashboardSummary(connected=True, syncing=False, trends=trends)
