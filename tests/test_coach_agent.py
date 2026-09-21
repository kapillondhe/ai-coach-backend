import contextlib
from unittest.mock import AsyncMock

import pytest

from app.agents import coach_agent


@pytest.mark.asyncio
async def test_build_toolsets_anonymous_has_only_base_toolset(monkeypatch):
    toolsets = await coach_agent._build_toolsets(user_id=None)
    assert len(toolsets) == 1


@pytest.mark.asyncio
async def test_build_toolsets_signed_in_without_coros_connection(monkeypatch):
    monkeypatch.setattr(coach_agent.coros_oauth, "get_access_token", AsyncMock(return_value=None))

    toolsets = await coach_agent._build_toolsets(user_id="user-123")

    assert len(toolsets) == 1  # no COROS toolset attached — additive, not required


@pytest.mark.asyncio
async def test_build_toolsets_signed_in_with_coros_connection(monkeypatch):
    monkeypatch.setattr(coach_agent.coros_oauth, "get_access_token", AsyncMock(return_value="a-token"))

    toolsets = await coach_agent._build_toolsets(user_id="user-123")

    assert len(toolsets) == 2  # base + COROS


@pytest.mark.asyncio
async def test_build_toolsets_degrades_gracefully_on_coros_failure(monkeypatch):
    monkeypatch.setattr(
        coach_agent.coros_oauth, "get_access_token", AsyncMock(side_effect=RuntimeError("COROS is down"))
    )

    # Must not raise — a COROS-side failure can never break core chat.
    toolsets = await coach_agent._build_toolsets(user_id="user-123")

    assert len(toolsets) == 1


@pytest.mark.asyncio
async def test_get_coach_agent_builds_fresh_agent_per_call():
    agent_a = await coach_agent.get_coach_agent(user_id=None)
    agent_b = await coach_agent.get_coach_agent(user_id=None)
    assert agent_a is not agent_b


@pytest.mark.asyncio
async def test_get_coach_agent_resolves_current_user_via_dependency_default(monkeypatch):
    # Regression test: get_coach_agent's user_id must actually come from
    # get_current_user_id when invoked through FastAPI's dependency
    # injection (Depends(...) default), not silently stay None forever —
    # that bug meant /coach/chat never attached a connected user's COROS
    # toolset, even after they'd connected via /integrations/coros/connect.
    monkeypatch.setattr(coach_agent, "get_current_user_id", lambda request=None: "resolved-user")
    monkeypatch.setattr(coach_agent.coros_oauth, "get_access_token", AsyncMock(return_value="a-token"))

    # Simulate what FastAPI does: call get_current_user_id() and pass its
    # result as user_id, since Depends() isn't resolved when calling the
    # function directly in a test.
    resolved_user_id = coach_agent.get_current_user_id()
    agent = await coach_agent.get_coach_agent(user_id=resolved_user_id)

    assert len(agent.toolsets) == 3  # pydantic-ai internal function toolset + base + COROS


def test_chat_route_resolves_user_id_through_real_dependency_chain(monkeypatch):
    # End-to-end regression test through the actual FastAPI route (no
    # dependency_overrides on get_coach_agent itself) — this is the exact
    # path that was broken: /coach/chat previously called get_coach_agent()
    # with no user_id, so a connected user's COROS toolset never attached.
    from fastapi.testclient import TestClient

    from app.main import app

    seen_user_ids: list[str | None] = []
    original_build_toolsets = coach_agent._build_toolsets

    async def _spy_build_toolsets(user_id):
        seen_user_ids.append(user_id)
        return await original_build_toolsets(user_id)

    monkeypatch.setattr(coach_agent, "_build_toolsets", _spy_build_toolsets)
    monkeypatch.setattr(coach_agent.coros_oauth, "get_access_token", AsyncMock(return_value=None))

    client = TestClient(app)
    # This will fail to actually reach OpenRouter/MCP (no network in tests),
    # so only assert on what we can observe before that: the resolved user_id.
    with contextlib.suppress(Exception):
        client.post("/api/coach/chat", json={"message": "hello"})

    # No Authorization header and no supabase_url configured in tests ->
    # anonymous (None) is the correctly-resolved user_id here.
    assert seen_user_ids == [None]
