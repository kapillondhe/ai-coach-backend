"""Dashboard service tests: verify the three response states (not connected,
syncing, connected with data) and trend-takeaway logic, all against a mocked
DB session — no live COROS/MCP calls here (that's covered by
tests/test_coros_sync_service.py and the live smoke test in #10).
"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.services import dashboard
from tests.fakes import FakeSession, stmt_kind, stmt_params


class _FakeSession(FakeSession):
    """Answers each SELECT by the table it reads (and metric_type/snapshot_type), not by call
    order, since get_dashboard_summary gathers its reads concurrently."""

    def __init__(self, data: dict, log: list):
        super().__init__()
        self.data = data
        self.log = log

    async def handle(self, stmt):
        assert stmt_kind(stmt) == "Select", f"Unexpected statement type: {stmt_kind(stmt)}"
        table = stmt.get_final_froms()[0].name
        params = stmt_params(stmt)
        key = table
        if table == "synced_daily_metrics":
            key = f"{table}:{params['metric_type_1']}"
        elif table == "synced_snapshots":
            key = f"{table}:{params['snapshot_type_1']}"
        self.log.append(key)
        value = self.data.get(key, [])
        return SimpleNamespace(first=lambda: value[0] if value else None, all=lambda: value)


def _sessions(data: dict, log: list | None = None):
    """Every get_session() call gets its own _FakeSession over the same per-table data."""
    log = [] if log is None else log
    return lambda: _FakeSession(data, log)


@pytest.mark.asyncio
async def test_get_dashboard_summary_not_connected(monkeypatch):
    monkeypatch.setattr(dashboard, "get_session", _sessions({}))

    summary = await dashboard.get_dashboard_summary("user-123")

    assert summary.connected is False
    assert summary.syncing is False
    assert summary.trends == []


@pytest.mark.asyncio
async def test_get_dashboard_summary_connected_but_syncing(monkeypatch):
    monkeypatch.setattr(
        dashboard,
        "get_session",
        _sessions(
            {
                "user_integrations": [SimpleNamespace(provider="coros")],
                "integration_sync_state": [
                    SimpleNamespace(
                        status="syncing", last_synced_at=None, last_backfill_completed_at=None, last_error=None
                    )
                ],
            }
        ),
    )

    summary = await dashboard.get_dashboard_summary("user-123")

    assert summary.connected is True
    assert summary.syncing is True


@pytest.mark.asyncio
async def test_get_dashboard_summary_connected_any_provider(monkeypatch):
    """Provider-generic on purpose: a non-COROS integration still counts as connected."""
    monkeypatch.setattr(
        dashboard, "get_session", _sessions({"user_integrations": [SimpleNamespace(provider="strava")]})
    )

    summary = await dashboard.get_dashboard_summary("user-123")

    assert summary.connected is True
    assert summary.syncing is True  # no sync state yet


_CONNECTED_WITH_DATA = {
    "user_integrations": [SimpleNamespace(provider="coros")],
    "integration_sync_state": [
        SimpleNamespace(
            status="idle",
            last_synced_at=datetime(2026, 9, 24, tzinfo=UTC),
            last_backfill_completed_at=datetime(2026, 9, 24, tzinfo=UTC),
            last_error=None,
        )
    ],
    "synced_daily_metrics:resting_hr": [
        SimpleNamespace(date="2026-09-21", value=49.0, extra={}),
        SimpleNamespace(date="2026-09-22", value=52.0, extra={}),
        SimpleNamespace(date="2026-09-23", value=53.0, extra={}),
        SimpleNamespace(date="2026-09-24", value=55.0, extra={}),
    ],
    "synced_daily_metrics:training_load": [
        SimpleNamespace(date="2026-09-24", value=0.55, extra={"comment": "Resuming", "short_term_load": 22.0})
    ],
    "synced_snapshots:fitness_assessment": [SimpleNamespace(data={"vo2max": "49", "threshold_pace": "5:32 /km"})],
}


@pytest.mark.asyncio
async def test_get_dashboard_summary_connected_with_data(monkeypatch):
    monkeypatch.setattr(dashboard, "get_session", _sessions(_CONNECTED_WITH_DATA))

    summary = await dashboard.get_dashboard_summary("user-123")

    assert summary.connected is True
    assert summary.syncing is False

    trend_ids = {t.id for t in summary.trends}
    assert trend_ids == {"resting_hr", "training_load", "fitness_assessment"}

    resting_trend = next(t for t in summary.trends if t.id == "resting_hr")
    assert resting_trend.latest_value == 55.0
    assert len(resting_trend.sparkline) == 4

    load_trend = next(t for t in summary.trends if t.id == "training_load")
    assert "Resuming" in load_trend.takeaway

    fitness_trend = next(t for t in summary.trends if t.id == "fitness_assessment")
    assert "49" in fitness_trend.takeaway


@pytest.mark.asyncio
async def test_get_dashboard_summary_runs_reads_concurrently(monkeypatch):
    """The three trend reads overlap instead of running one after another."""
    in_flight = 0
    peak = 0
    real_sessions = _sessions(_CONNECTED_WITH_DATA)

    class _SlowSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.inner = real_sessions()

        async def handle(self, stmt):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return await self.inner.execute(stmt)

    monkeypatch.setattr(dashboard, "get_session", _SlowSession)

    summary = await dashboard.get_dashboard_summary("user-123")

    assert len(summary.trends) == 3
    assert peak == 3


@pytest.mark.asyncio
async def test_get_dashboard_summary_window_uses_utc_date(monkeypatch):
    log: list = []
    captured: list = []
    real = dashboard._daily_metric_series

    async def _spy(user_id, metric_type, since):
        captured.append(since)
        return await real(user_id, metric_type, since)

    monkeypatch.setattr(dashboard, "_daily_metric_series", _spy)
    monkeypatch.setattr(dashboard, "get_session", _sessions(_CONNECTED_WITH_DATA, log))

    await dashboard.get_dashboard_summary("user-123")

    expected = datetime.now(UTC).date() - timedelta(weeks=dashboard.TRAILING_WEEKS)
    assert captured == [expected, expected]


def test_resting_hr_takeaway_not_enough_data():
    assert "Not enough" in dashboard._resting_hr_takeaway([("2026-09-24", 50.0, {})])


def test_resting_hr_takeaway_trending_up():
    series = [(f"2026-09-{i:02d}", float(40 + i), {}) for i in range(1, 9)]
    takeaway = dashboard._resting_hr_takeaway(series)
    assert "up" in takeaway


def test_resting_hr_takeaway_stable():
    series = [(f"2026-09-{i:02d}", 50.0, {}) for i in range(1, 9)]
    assert dashboard._resting_hr_takeaway(series) == "Resting heart rate has been stable."


def test_training_load_takeaway_uses_comment_and_ratio():
    series = [("2026-09-24", 0.55, {"comment": "Resuming"})]
    takeaway = dashboard._training_load_takeaway(series)
    assert "Resuming" in takeaway
    assert "0.55" in takeaway


def test_training_load_takeaway_empty():
    assert "Not enough" in dashboard._training_load_takeaway([])


def test_fitness_takeaway_no_snapshot():
    assert "Connect a device" in dashboard._fitness_takeaway(None)


def test_fitness_takeaway_with_snapshot():
    takeaway = dashboard._fitness_takeaway({"vo2max": "49", "threshold_pace": "5:32 /km"})
    assert "49" in takeaway
    assert "5:32" in takeaway
