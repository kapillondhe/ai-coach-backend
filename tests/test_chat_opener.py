from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.models import Discipline
from app.services import chat_opener
from tests.fakes import ScriptedSession


def _sessions(*result_sets):
    iterator = iter(result_sets)

    def _factory():
        return ScriptedSession(next(iterator))

    return _factory


@pytest.mark.asyncio
async def test_anonymous_gets_static_opener():
    opener = await chat_opener.get_chat_opener(None)

    assert opener.text == chat_opener._ANONYMOUS_OPENER
    assert opener.chips == chat_opener._ANONYMOUS_CHIPS


@pytest.mark.asyncio
async def test_signed_in_not_connected_gets_static_opener(monkeypatch):
    is_connected = AsyncMock(return_value=False)
    monkeypatch.setattr(chat_opener.coros_oauth, "is_connected", is_connected)
    monkeypatch.setattr(chat_opener, "get_session", _sessions())  # must not be reached

    opener = await chat_opener.get_chat_opener("user-123")

    assert opener.text == chat_opener._SIGNED_IN_OPENER
    assert opener.chips == chat_opener._SIGNED_IN_CHIPS
    is_connected.assert_awaited_once_with("user-123")


@pytest.mark.asyncio
async def test_signed_in_connected_no_activity_falls_back(monkeypatch):
    monkeypatch.setattr(chat_opener.coros_oauth, "is_connected", AsyncMock(return_value=True))
    monkeypatch.setattr(chat_opener, "get_session", _sessions(None))  # _latest_activity -> none

    opener = await chat_opener.get_chat_opener("user-123")

    assert opener.text == chat_opener._SIGNED_IN_OPENER


@pytest.mark.asyncio
async def test_signed_in_connected_with_activity_references_it(monkeypatch):
    # Within the last week, relative to whenever the test runs, so this doesn't
    # silently become a "weeks ago" assertion as time passes.
    started_at = datetime.now(UTC) - timedelta(days=2)
    activity = SimpleNamespace(
        discipline=Discipline.BIKE,
        started_at=started_at,
        distance_km=62.3,
        duration_seconds=7200,
    )
    monkeypatch.setattr(chat_opener.coros_oauth, "is_connected", AsyncMock(return_value=True))
    # _latest_activity result set: pop(0) -> activity, .scalars().first() -> activity
    monkeypatch.setattr(chat_opener, "get_session", _sessions([activity]))

    opener = await chat_opener.get_chat_opener("user-123")

    assert "ride" in opener.text
    assert started_at.strftime("%A") in opener.text
    assert "62.3km" in opener.text
    assert opener.chips == chat_opener._CONNECTED_CHIPS


def test_describe_activity_handles_missing_distance_and_duration():
    activity = SimpleNamespace(discipline="run", started_at=None, distance_km=None, duration_seconds=None)

    description = chat_opener._describe_activity(activity)

    assert description.startswith("Nice run recently")


@pytest.mark.parametrize(
    ("days_ago", "expected_substring"),
    [
        (0, "today"),
        (1, "yesterday"),
        (13, "about a week ago"),
        (20, "about 2 weeks ago"),
        (90, "a while ago"),
    ],
)
def test_describe_when_reflects_real_elapsed_time(days_ago, expected_substring):
    started_at = datetime.now(UTC) - timedelta(days=days_ago)

    assert chat_opener._describe_when(started_at) == expected_substring


def test_describe_when_within_the_week_names_the_weekday():
    started_at = datetime.now(UTC) - timedelta(days=4)

    assert chat_opener._describe_when(started_at) == f"on {started_at.strftime('%A')}"


def test_describe_when_none_is_recently():
    assert chat_opener._describe_when(None) == "recently"


@pytest.mark.parametrize(
    ("discipline", "word"),
    [("run", "run"), ("bike", "ride"), ("swim", "swim"), ("other", "session"), ("unknown", "session")],
)
def test_describe_activity_maps_db_discipline_strings(discipline, word):
    # Rows come back from the TEXT column as plain strings; StrEnum keys must still match.
    activity = SimpleNamespace(discipline=discipline, started_at=None, distance_km=None, duration_seconds=None)

    assert chat_opener._describe_activity(activity).startswith(f"Nice {word} recently")
