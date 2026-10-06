"""Service-level tests for the COROS sync layer's non-parser logic: idempotent
upsert behavior and periodic-sync resilience to a single user's failure.

The MCP session and DB layer are mocked here — MCP text-response parsing is
covered separately (test_coros_sync_parsers.py) against real captured
COROS output.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.models import Discipline, SyncStatusValue
from app.services import coros_sync
from app.services import coros_sync_parsers as parsers
from tests.fakes import FakeSession, multirow_values, stmt_kind


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_mcp_call_tool_text(responses: dict[str, str]):
    """Patch coros_sync._call_tool_text to return canned text per tool name."""

    async def _call(session, name, arguments):
        return responses.get(name, "")

    return _call


@pytest.mark.asyncio
async def test_sync_user_raises_when_not_connected(monkeypatch):
    monkeypatch.setattr(coros_sync.coros_oauth, "get_access_token", AsyncMock(return_value=None))

    with pytest.raises(coros_sync.CorosSyncError, match="no active COROS connection"):
        await coros_sync.sync_user("user-123", days=28)


@pytest.mark.asyncio
async def test_sync_user_stores_parsed_data(monkeypatch):
    monkeypatch.setattr(coros_sync.coros_oauth, "get_access_token", AsyncMock(return_value="token"))

    @asynccontextmanager
    async def _fake_session(access_token):
        yield object()

    monkeypatch.setattr(coros_sync, "_mcp_session", _fake_session)
    monkeypatch.setattr(
        coros_sync,
        "_call_tool_text",
        _fake_mcp_call_tool_text(
            {
                "querySportRecords": (
                    "1. Outdoor Run — 2026-09-21\n"
                    "Time Window: startTimestamp=1789996097 | endTimestamp=1789997859\n"
                    "Duration: 29:22 | Distance: 4.02 km\n"
                    "Average Pace: 7:18 /km | Avg HR: 151 bpm | Calories: 453 kcal\n"
                    "LabelId: abc123 | SportType: 100\n"
                ),
                "queryRestingHeartRate": "2026-09-21: 49 bpm",
                "queryAvgHeartRate": "2026-09-21: 60 bpm (Min: 37, Max: 171)",
                "queryTrainingLoadAssessment": (
                    "2026-09-21\nComment: Resuming\nShort-Term Load: 32\nLong-Term Load: 42\nLoad Ratio: 0.76"
                ),
                "queryRecoveryStatus": "Recovery: 100%\nLevel: Heavy training allowed",
                "queryFitnessAssessmentOverview": "VO2max: 49",
            }
        ),
    )

    set_status = AsyncMock()
    mark_synced = AsyncMock()
    upsert_activities = AsyncMock(return_value=1)
    upsert_daily = AsyncMock(return_value=1)
    insert_snapshot = AsyncMock()
    monkeypatch.setattr(coros_sync, "_set_sync_status", set_status)
    monkeypatch.setattr(coros_sync, "_mark_synced", mark_synced)
    monkeypatch.setattr(coros_sync, "_upsert_activities", upsert_activities)
    monkeypatch.setattr(coros_sync, "_upsert_daily_metrics", upsert_daily)
    monkeypatch.setattr(coros_sync, "_insert_snapshot", insert_snapshot)

    await coros_sync.sync_user("user-123", days=28, backfill=True)

    upsert_activities.assert_awaited_once()
    assert upsert_daily.await_count == 3  # resting_hr, avg_hr, training_load
    assert insert_snapshot.await_count == 2  # recovery_status, fitness_assessment
    mark_synced.assert_awaited_once_with("user-123", backfill=True)
    set_status.assert_awaited_once_with("user-123", status=SyncStatusValue.SYNCING)


@pytest.mark.asyncio
async def test_sync_user_marks_error_status_on_mcp_failure(monkeypatch):
    monkeypatch.setattr(coros_sync.coros_oauth, "get_access_token", AsyncMock(return_value="token"))

    @asynccontextmanager
    async def _raising_session(access_token):
        raise ConnectionError("COROS unreachable")
        yield  # pragma: no cover — unreachable, satisfies generator shape

    monkeypatch.setattr(coros_sync, "_mcp_session", _raising_session)
    set_status = AsyncMock()
    monkeypatch.setattr(coros_sync, "_set_sync_status", set_status)

    with pytest.raises(coros_sync.CorosSyncError):
        await coros_sync.sync_user("user-123", days=28)

    set_status.assert_any_call("user-123", status="syncing")  # StrEnum compares equal to the raw string
    error_calls = [c for c in set_status.await_args_list if c.kwargs.get("status") == SyncStatusValue.ERROR]
    assert len(error_calls) == 1


def _patch_sync_happy_path(monkeypatch, calls: dict | None = None):
    """Stub everything sync_user touches; record the querySportRecords args in `calls`."""
    monkeypatch.setattr(coros_sync.coros_oauth, "get_access_token", AsyncMock(return_value="token"))

    @asynccontextmanager
    async def _fake_session(access_token):
        yield object()

    async def _call(session, name, arguments):
        if calls is not None:
            calls[name] = arguments
        return ""

    monkeypatch.setattr(coros_sync, "_mcp_session", _fake_session)
    monkeypatch.setattr(coros_sync, "_call_tool_text", _call)
    monkeypatch.setattr(coros_sync, "_set_sync_status", AsyncMock())
    mark_synced = AsyncMock()
    monkeypatch.setattr(coros_sync, "_mark_synced", mark_synced)
    return mark_synced


@pytest.mark.asyncio
async def test_sync_user_backfill_window_without_flag_is_not_a_backfill(monkeypatch):
    """Backfill is an explicit flag now, not inferred from days == BACKFILL_WINDOW_DAYS."""
    mark_synced = _patch_sync_happy_path(monkeypatch)

    await coros_sync.sync_user("user-123", days=coros_sync.BACKFILL_WINDOW_DAYS)

    mark_synced.assert_awaited_once_with("user-123", backfill=False)


@pytest.mark.asyncio
async def test_sync_user_date_window_is_utc(monkeypatch):
    calls: dict = {}
    _patch_sync_happy_path(monkeypatch, calls)

    await coros_sync.sync_user("user-123", days=7)

    today = datetime.now(UTC).date()
    assert calls["querySportRecords"]["endDate"] == today.strftime("%Y%m%d")
    assert calls["querySportRecords"]["startDate"] == (today - timedelta(days=7)).strftime("%Y%m%d")


@pytest.mark.asyncio
async def test_backfill_user_passes_backfill_flag(monkeypatch):
    sync_user = AsyncMock()
    monkeypatch.setattr(coros_sync, "sync_user", sync_user)

    await coros_sync.backfill_user("user-123")

    sync_user.assert_awaited_once_with("user-123", days=coros_sync.BACKFILL_WINDOW_DAYS, backfill=True)


@pytest.mark.asyncio
async def test_backfill_user_never_raises(monkeypatch):
    monkeypatch.setattr(coros_sync, "sync_user", AsyncMock(side_effect=coros_sync.CorosSyncError("boom")))
    await coros_sync.backfill_user("user-123")  # should not raise


@pytest.mark.asyncio
async def test_backfill_user_swallows_non_sync_errors(monkeypatch):
    monkeypatch.setattr(coros_sync, "sync_user", AsyncMock(side_effect=OSError("DB unreachable")))
    await coros_sync.backfill_user("user-123")  # should not raise


@pytest.mark.asyncio
async def test_run_periodic_sync_continues_past_one_users_failure(monkeypatch):
    class _FakeSession(FakeSession):
        async def handle(self, stmt):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: ["user-1", "user-2"]))

    monkeypatch.setattr(coros_sync, "get_session", lambda: _FakeSession())

    calls = []

    async def _sync_user(user_id, *, days, backfill=False):
        calls.append((user_id, days, backfill))
        if user_id == "user-1":
            raise coros_sync.CorosSyncError("user-1 token expired")
        return None

    monkeypatch.setattr(coros_sync, "sync_user", _sync_user)

    result = await coros_sync.run_periodic_sync()

    assert result == {"synced": 1, "failed": 1}
    assert calls == [
        ("user-1", coros_sync.PERIODIC_SYNC_WINDOW_DAYS, False),
        ("user-2", coros_sync.PERIODIC_SYNC_WINDOW_DAYS, False),
    ]


class _FakeUpsertSession(FakeSession):
    """Records each pg_insert(...).on_conflict_do_update with its rows decoded."""

    def __init__(self, log: list):
        super().__init__()
        self.log = log

    async def handle(self, stmt):
        if stmt_kind(stmt) != "Insert":
            return await super().handle(stmt)
        rows = multirow_values(stmt)
        self.log.append(
            SimpleNamespace(table=stmt.table.name, rows=rows, on_conflict=stmt._post_values_clause is not None)
        )
        return SimpleNamespace(rowcount=len(rows))


def _activity(external_id: str, *, calories: int = 100) -> parsers.ParsedActivity:
    return parsers.ParsedActivity(
        external_id=external_id,
        discipline=Discipline.RUN,
        sport_type_code=100,
        started_at_epoch=1789996097,
        ended_at_epoch=1789997859,
        duration_seconds=1762,
        distance_km=4.02,
        avg_pace_sec_per_km=438,
        avg_hr=151,
        calories=calories,
        raw_text="raw",
    )


@pytest.mark.asyncio
async def test_upsert_activities_uses_one_multi_row_statement(monkeypatch):
    log: list = []
    monkeypatch.setattr(coros_sync, "get_session", lambda: _FakeUpsertSession(log))

    count = await coros_sync._upsert_activities("user-123", [_activity("a"), _activity("b"), _activity("c")])

    assert count == 3
    assert len(log) == 1
    (stmt,) = log
    assert stmt.table == "synced_activities"
    assert stmt.on_conflict
    assert [r["external_id"] for r in stmt.rows] == ["a", "b", "c"]
    assert {r["discipline"] for r in stmt.rows} == {"run"}
    assert stmt.rows[0]["provider"] == "coros"
    assert stmt.rows[0]["started_at"] == datetime.fromtimestamp(1789996097, tz=UTC)


@pytest.mark.asyncio
async def test_upsert_activities_keeps_last_duplicate_external_id(monkeypatch):
    """One multi-row ON CONFLICT can't touch the same row twice; keep the last, like the old loop."""
    log: list = []
    monkeypatch.setattr(coros_sync, "get_session", lambda: _FakeUpsertSession(log))

    count = await coros_sync._upsert_activities(
        "user-123", [_activity("a", calories=1), _activity("b"), _activity("a", calories=2)]
    )

    assert count == 2
    rows = {r["external_id"]: r for r in log[0].rows}
    assert rows["a"]["calories"] == 2


@pytest.mark.asyncio
async def test_upsert_activities_chunks_large_batches(monkeypatch):
    log: list = []
    sessions: list = []

    def _factory():
        sessions.append(_FakeUpsertSession(log))
        return sessions[-1]

    monkeypatch.setattr(coros_sync, "get_session", _factory)
    monkeypatch.setattr(coros_sync, "UPSERT_BATCH_SIZE", 2)

    count = await coros_sync._upsert_activities("user-123", [_activity(str(i)) for i in range(5)])

    assert count == 5
    assert [len(s.rows) for s in log] == [2, 2, 1]
    assert len(sessions) == 1 and sessions[0].commits == 1  # one session, one commit


@pytest.mark.asyncio
async def test_upsert_daily_metrics_uses_one_multi_row_statement(monkeypatch):
    log: list = []
    monkeypatch.setattr(coros_sync, "get_session", lambda: _FakeUpsertSession(log))
    values = [
        parsers.ParsedDailyValue(date="2026-09-20", value=48.0),
        parsers.ParsedDailyValue(date="2026-09-21", value=None),
        parsers.ParsedDailyValue(date="2026-09-20", value=49.0),  # duplicate date: last wins
    ]

    count = await coros_sync._upsert_daily_metrics("user-123", "resting_hr", values)

    assert count == 2
    assert len(log) == 1
    (stmt,) = log
    assert stmt.table == "synced_daily_metrics"
    assert stmt.on_conflict
    assert {r["date"]: r["value"] for r in stmt.rows} == {"2026-09-20": 49.0, "2026-09-21": None}
    assert {r["metric_type"] for r in stmt.rows} == {"resting_hr"}


@pytest.mark.asyncio
async def test_upserts_skip_db_when_empty(monkeypatch):
    def _no_session():
        raise AssertionError("no DB session expected for an empty batch")

    monkeypatch.setattr(coros_sync, "get_session", _no_session)

    assert await coros_sync._upsert_activities("user-123", []) == 0
    assert await coros_sync._upsert_daily_metrics("user-123", "avg_hr", []) == 0


@pytest.mark.asyncio
async def test_get_sync_status_returns_typed_status(monkeypatch):
    row = SimpleNamespace(status="error", last_synced_at=None, last_backfill_completed_at=None, last_error="boom")

    class _Session(FakeSession):
        async def handle(self, stmt):
            assert stmt_kind(stmt) == "Select"
            return SimpleNamespace(first=lambda: row)

    monkeypatch.setattr(coros_sync, "get_session", _Session)

    status = await coros_sync.get_sync_status("user-123")

    assert status.status is SyncStatusValue.ERROR
    assert status.status == "error"
