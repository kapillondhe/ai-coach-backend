"""Dashboard service tests: verify the three response states (not connected,
syncing, connected with data) and trend-takeaway logic, all against a mocked
DB session — no live COROS/MCP calls here (that's covered by
tests/test_coros_sync_service.py and the live smoke test in #10).
"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.services import dashboard


class _FakeSession:
    """Scripted results returned in call order, mirroring test_coros_oauth.py's pattern."""

    def __init__(self, results=None):
        self.results = list(results or [])

    async def execute(self, stmt):
        value = self.results.pop(0) if self.results else []
        return SimpleNamespace(
            first=lambda: value if not isinstance(value, list) else (value[0] if value else None),
            all=lambda: value if isinstance(value, list) else [value],
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _sessions(*result_sets):
    """Each get_session() call in the service gets its own _FakeSession with one result set."""
    iterator = iter(result_sets)

    def _factory():
        return _FakeSession(next(iterator))

    return _factory


@pytest.mark.asyncio
async def test_get_dashboard_summary_not_connected(monkeypatch):
    monkeypatch.setattr(dashboard, "get_session", _sessions([None]))

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
            [SimpleNamespace(provider="coros")],  # _is_connected
            [
                SimpleNamespace(
                    status="syncing",
                    last_synced_at=None,
                    last_backfill_completed_at=None,
                    last_error=None,
                )
            ],  # _get_any_sync_state
        ),
    )

    summary = await dashboard.get_dashboard_summary("user-123")

    assert summary.connected is True
    assert summary.syncing is True


@pytest.mark.asyncio
async def test_get_dashboard_summary_connected_with_data(monkeypatch):
    now = datetime(2026, 9, 24, tzinfo=UTC)
    monkeypatch.setattr(
        dashboard,
        "get_session",
        _sessions(
            [SimpleNamespace(provider="coros")],  # _is_connected
            [
                SimpleNamespace(
                    status="idle", last_synced_at=now, last_backfill_completed_at=now, last_error=None
                )
            ],  # _get_any_sync_state
            [
                SimpleNamespace(date="2026-09-21", value=49.0, extra={}),
                SimpleNamespace(date="2026-09-22", value=52.0, extra={}),
                SimpleNamespace(date="2026-09-23", value=53.0, extra={}),
                SimpleNamespace(date="2026-09-24", value=55.0, extra={}),
            ],  # resting_hr series
            [
                SimpleNamespace(
                    date="2026-09-24", value=0.55, extra={"comment": "Resuming", "short_term_load": 22.0}
                )
            ],  # training_load series
            [SimpleNamespace(data={"vo2max": "49", "threshold_pace": "5:32 /km"})],  # fitness snapshot
        ),
    )

    summary = await dashboard.get_dashboard_summary("user-123")

    assert summary.connected is True
    assert summary.syncing is False

    trend_ids = {t.id for t in summary.trends}
    assert trend_ids == {"resting_hr", "training_load", "fitness_assessment"}

    load_trend = next(t for t in summary.trends if t.id == "training_load")
    assert "Resuming" in load_trend.takeaway

    fitness_trend = next(t for t in summary.trends if t.id == "fitness_assessment")
    assert "49" in fitness_trend.takeaway


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
