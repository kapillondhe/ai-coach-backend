"""Service-level tests for the COROS sync layer's non-parser logic: idempotent
upsert behavior and periodic-sync resilience to a single user's failure.

The MCP session and DB layer are mocked here — MCP text-response parsing is
covered separately (test_coros_sync_parsers.py) against real captured
COROS output.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app.services import coros_sync


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

    await coros_sync.sync_user("user-123", days=28)

    upsert_activities.assert_awaited_once()
    assert upsert_daily.await_count == 3  # resting_hr, avg_hr, training_load
    assert insert_snapshot.await_count == 2  # recovery_status, fitness_assessment
    mark_synced.assert_awaited_once_with("user-123", backfill=True)


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

    set_status.assert_any_call("user-123", status="syncing")
    error_calls = [c for c in set_status.await_args_list if c.kwargs.get("status") == "error"]
    assert len(error_calls) == 1


@pytest.mark.asyncio
async def test_backfill_user_never_raises(monkeypatch):
    monkeypatch.setattr(
        coros_sync, "sync_user", AsyncMock(side_effect=coros_sync.CorosSyncError("boom"))
    )
    await coros_sync.backfill_user("user-123")  # should not raise


@pytest.mark.asyncio
async def test_run_periodic_sync_continues_past_one_users_failure(monkeypatch):
    class _FakeScalars:
        def __init__(self, values):
            self._values = values

        def all(self):
            return self._values

    class _FakeResult:
        def __init__(self, values):
            self._values = values

        def scalars(self):
            return _FakeScalars(self._values)

    class _FakeSession:
        async def execute(self, stmt):
            return _FakeResult(["user-1", "user-2"])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(coros_sync, "get_session", lambda: _FakeSession())

    async def _sync_user(user_id, *, days):
        if user_id == "user-1":
            raise coros_sync.CorosSyncError("user-1 token expired")
        return None

    monkeypatch.setattr(coros_sync, "sync_user", _sync_user)

    result = await coros_sync.run_periodic_sync()

    assert result == {"synced": 1, "failed": 1}
