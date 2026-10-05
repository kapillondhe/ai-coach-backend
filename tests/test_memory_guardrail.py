"""Unit tests for app.services.memory_guardrail — the memory-save classifier.

Mocks pydantic_ai.Agent.run rather than hitting the real Jev/OpenRouter API,
mirroring tests/test_scope_service.py's convention for the same transport.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import memory_guardrail


@pytest.mark.asyncio
async def test_is_worth_remembering_true_when_classifier_says_durable():
    with patch.object(memory_guardrail.Agent, "run", AsyncMock(return_value=SimpleNamespace(output=True))):
        assert await memory_guardrail.is_worth_remembering("Training for a first 70.3 in June") is True


@pytest.mark.asyncio
async def test_is_worth_remembering_false_when_classifier_says_trivial():
    with patch.object(memory_guardrail.Agent, "run", AsyncMock(return_value=SimpleNamespace(output=False))):
        assert await memory_guardrail.is_worth_remembering("ok thanks") is False


@pytest.mark.asyncio
async def test_is_worth_remembering_fails_open_on_classifier_error():
    """A transient model/API failure must never silently drop a candidate fact."""
    with patch.object(memory_guardrail.Agent, "run", AsyncMock(side_effect=RuntimeError("model unavailable"))):
        assert await memory_guardrail.is_worth_remembering("Has a history of runner's knee") is True


@pytest.mark.asyncio
async def test_is_worth_remembering_includes_fact_in_prompt():
    run_mock = AsyncMock(return_value=SimpleNamespace(output=True))
    with patch.object(memory_guardrail.Agent, "run", run_mock):
        await memory_guardrail.is_worth_remembering("Training for a first 70.3 in June")

    args, _ = run_mock.await_args
    assert args[0] == "FACT: Training for a first 70.3 in June"
