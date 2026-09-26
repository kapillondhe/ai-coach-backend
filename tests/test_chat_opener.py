from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.services import chat_opener


class _FakeSession:
    """Scripted results returned in call order, mirroring test_dashboard_service.py's pattern."""

    def __init__(self, results=None):
        self.results = list(results or [])

    async def execute(self, stmt):
        value = self.results.pop(0) if self.results else None
        return SimpleNamespace(first=lambda: value, scalars=lambda: SimpleNamespace(first=lambda: value))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _sessions(*result_sets):
    iterator = iter(result_sets)

    def _factory():
        return _FakeSession(next(iterator))

    return _factory


@pytest.mark.asyncio
async def test_anonymous_gets_static_opener():
    opener = await chat_opener.get_chat_opener(None)

    assert opener.text == chat_opener._ANONYMOUS_OPENER
    assert opener.chips == chat_opener._ANONYMOUS_CHIPS


@pytest.mark.asyncio
async def test_signed_in_not_connected_gets_static_opener(monkeypatch):
    monkeypatch.setattr(chat_opener, "get_session", _sessions([None]))  # _is_coros_connected -> False

    opener = await chat_opener.get_chat_opener("user-123")

    assert opener.text == chat_opener._SIGNED_IN_OPENER
    assert opener.chips == chat_opener._SIGNED_IN_CHIPS


@pytest.mark.asyncio
async def test_signed_in_connected_no_activity_falls_back(monkeypatch):
    monkeypatch.setattr(
        chat_opener,
        "get_session",
        _sessions(
            [SimpleNamespace(provider="coros")],  # _is_coros_connected -> True
            None,  # _latest_activity -> none
        ),
    )

    opener = await chat_opener.get_chat_opener("user-123")

    assert opener.text == chat_opener._SIGNED_IN_OPENER


@pytest.mark.asyncio
async def test_signed_in_connected_with_activity_references_it(monkeypatch):
    activity = SimpleNamespace(
        discipline="bike",
        started_at=datetime(2026, 9, 26, tzinfo=UTC),  # a Saturday
        distance_km=62.3,
        duration_seconds=7200,
    )
    monkeypatch.setattr(
        chat_opener,
        "get_session",
        _sessions(
            [SimpleNamespace(provider="coros")],  # _is_coros_connected -> True
            [activity],  # _latest_activity result set: pop(0) -> activity, .scalars().first() -> activity
        ),
    )

    opener = await chat_opener.get_chat_opener("user-123")

    assert "ride" in opener.text
    assert "Saturday" in opener.text
    assert "62.3km" in opener.text
    assert opener.chips == chat_opener._CONNECTED_CHIPS


def test_describe_activity_handles_missing_distance_and_duration():
    activity = SimpleNamespace(discipline="run", started_at=None, distance_km=None, duration_seconds=None)

    description = chat_opener._describe_activity(activity)

    assert description.startswith("Nice run on recently")
