"""Unit tests for app.services.scope — the off-topic classifier guardrail.

Mocks pydantic_ai.Agent.run rather than hitting a real model, mirroring the
`get_coach_agent` tests' convention of not making real network calls in this
test file's scope. The classifier's actual accuracy against the live model was
verified manually (see the ai-coach-backend-patterns skill notes), not here.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.services import scope


@pytest.mark.asyncio
async def test_is_off_topic_true_when_classifier_says_off_topic():
    with patch.object(scope.Agent, "run", AsyncMock(return_value=SimpleNamespace(output=False))):
        assert await scope.is_off_topic("who's the best actor working today?") is True


@pytest.mark.asyncio
async def test_is_off_topic_false_when_classifier_says_on_topic():
    with patch.object(scope.Agent, "run", AsyncMock(return_value=SimpleNamespace(output=True))):
        assert await scope.is_off_topic("how much protein should I eat?") is False


@pytest.mark.asyncio
async def test_is_off_topic_fails_open_on_classifier_error():
    """A transient model/API failure must never block a legitimate question."""
    with patch.object(scope.Agent, "run", AsyncMock(side_effect=RuntimeError("model unavailable"))):
        assert await scope.is_off_topic("how much protein should I eat?") is False


@pytest.mark.asyncio
async def test_is_off_topic_flattens_history_into_prompt_text():
    """Regression test: history must NOT be passed as real pydantic-ai message_history —

    that was tried and regressed (the model got confused about the generic
    `final_result` boolean output tool with real history attached and started
    defaulting to `true`). It must be flattened into the single string prompt
    instead, with the latest message clearly marked.
    """
    run_mock = AsyncMock(return_value=SimpleNamespace(output=True))
    history = [
        ModelRequest(parts=[UserPromptPart(content="my name is Alex")]),
        ModelResponse(parts=[TextPart(content="Nice to meet you, Alex!")]),
    ]
    with patch.object(scope.Agent, "run", run_mock):
        await scope.is_off_topic("what's my name?", message_history=history)

    run_mock.assert_awaited_once()
    args, kwargs = run_mock.await_args
    assert "message_history" not in kwargs  # not passed as real history
    prompt = args[0]
    assert isinstance(prompt, str)
    assert "my name is Alex" in prompt
    assert "Nice to meet you, Alex!" in prompt
    assert "LATEST MESSAGE TO CLASSIFY: what's my name?" in prompt


@pytest.mark.asyncio
async def test_is_off_topic_with_no_history_still_marks_latest_message():
    run_mock = AsyncMock(return_value=SimpleNamespace(output=True))
    with patch.object(scope.Agent, "run", run_mock):
        await scope.is_off_topic("how much protein should I eat?")

    args, _ = run_mock.await_args
    assert args[0] == "LATEST MESSAGE TO CLASSIFY: how much protein should I eat?"
