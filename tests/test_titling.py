from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import titling


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Marathon Taper Plan", "Marathon Taper Plan"),
        ('"Quoted Title"', "Quoted Title"),
        ("First line\nsecond line", "First line"),
        ("   \n  ", ""),
        ("", ""),
        ("x" * 200, "x" * titling._MAX_TITLE_LENGTH),
    ],
)
def test_clean_title(raw, expected):
    assert titling._clean_title(raw) == expected


async def test_blank_model_output_sets_no_title(monkeypatch):
    set_title = AsyncMock()
    monkeypatch.setattr(titling.conversation_service, "set_title", set_title)
    monkeypatch.setattr(titling, "get_utility_model", lambda: "test")
    with (
        patch.object(titling.Agent, "run", AsyncMock(return_value=SimpleNamespace(output="  \n "))),
        patch.object(titling.logger, "exception") as log_exception,
    ):
        await titling.generate_and_set_title("conv-1", "user-1", "hi")

    set_title.assert_not_awaited()
    log_exception.assert_not_called()  # blank output is not an error


async def test_title_is_saved(monkeypatch):
    set_title = AsyncMock()
    monkeypatch.setattr(titling.conversation_service, "set_title", set_title)
    monkeypatch.setattr(titling, "get_utility_model", lambda: "test")
    with patch.object(titling.Agent, "run", AsyncMock(return_value=SimpleNamespace(output="Taper Advice"))):
        await titling.generate_and_set_title("conv-1", "user-1", "how do I taper?")

    set_title.assert_awaited_once_with("conv-1", "user-1", "Taper Advice")
