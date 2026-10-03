import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import ModelRetry

from app.services import coros_mcp


class _FakeMCPToolset:
    """Stands in for pydantic_ai's MCPToolset: tracks opens/closes and list_tools calls."""

    instances: list["_FakeMCPToolset"] = []

    def __init__(self, token: str) -> None:
        self.token = token
        self.entered = 0
        self.exited = 0
        _FakeMCPToolset.instances.append(self)

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *args):
        self.exited += 1

    async def list_tools(self):
        return []

    async def get_tools(self, ctx):
        return {}

    async def call_tool(self, name, tool_args, ctx, tool):
        raise NotImplementedError

    def apply(self, visitor):
        visitor(self)


@pytest.fixture(autouse=True)
def _fake_open(monkeypatch):
    _FakeMCPToolset.instances = []
    opens: list[str] = []

    async def _open(token):
        opens.append(token)
        toolset = _FakeMCPToolset(token)
        await toolset.__aenter__()
        return toolset

    monkeypatch.setattr(coros_mcp, "_open", _open)
    coros_mcp._sessions.clear()
    coros_mcp._locks.clear()
    yield opens
    coros_mcp._sessions.clear()
    coros_mcp._locks.clear()


async def _settle():
    # Let background close tasks run.
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_reuses_open_session_across_turns(_fake_open):
    first = await coros_mcp.get_toolset("u1", "tok")
    second = await coros_mcp.get_toolset("u1", "tok")

    assert _fake_open == ["tok"]  # connected once
    assert first.wrapped is second.wrapped


@pytest.mark.asyncio
async def test_sessions_are_per_user(_fake_open):
    a = await coros_mcp.get_toolset("u1", "tok-1")
    b = await coros_mcp.get_toolset("u2", "tok-2")

    assert a.wrapped is not b.wrapped
    assert _fake_open == ["tok-1", "tok-2"]


@pytest.mark.asyncio
async def test_concurrent_first_turns_open_only_one_session(_fake_open):
    results = await asyncio.gather(*(coros_mcp.get_toolset("u1", "tok") for _ in range(5)))

    assert _fake_open == ["tok"]
    assert len({id(r.wrapped) for r in results}) == 1


@pytest.mark.asyncio
async def test_new_token_replaces_and_closes_old_session(_fake_open):
    old = (await coros_mcp.get_toolset("u1", "old-tok")).wrapped
    new = (await coros_mcp.get_toolset("u1", "new-tok")).wrapped
    await _settle()

    assert new is not old
    assert new.token == "new-tok"
    assert old.exited == 1


@pytest.mark.asyncio
async def test_idle_session_is_reopened(_fake_open, monkeypatch):
    old = (await coros_mcp.get_toolset("u1", "tok")).wrapped
    monkeypatch.setattr(coros_mcp, "IDLE_TTL", -1.0)

    new = (await coros_mcp.get_toolset("u1", "tok")).wrapped
    await _settle()

    assert new is not old
    assert old.exited == 1


@pytest.mark.asyncio
async def test_failed_tool_call_evicts_session_so_next_turn_reconnects(_fake_open):
    toolset = await coros_mcp.get_toolset("u1", "tok")
    toolset.wrapped.call_tool = AsyncMock(side_effect=ConnectionError("session dropped"))

    with pytest.raises(ConnectionError):
        await toolset.call_tool("queryDevices", {}, None, None)
    await _settle()

    assert toolset.wrapped.exited == 1
    again = await coros_mcp.get_toolset("u1", "tok")
    assert again.wrapped is not toolset.wrapped
    assert _fake_open == ["tok", "tok"]


@pytest.mark.asyncio
async def test_model_retry_from_a_tool_does_not_evict(_fake_open):
    toolset = await coros_mcp.get_toolset("u1", "tok")
    toolset.wrapped.call_tool = AsyncMock(side_effect=ModelRetry("bad args"))

    with pytest.raises(ModelRetry):
        await toolset.call_tool("queryDevices", {}, None, None)

    again = await coros_mcp.get_toolset("u1", "tok")
    assert again.wrapped is toolset.wrapped


@pytest.mark.asyncio
async def test_open_failure_propagates_and_caches_nothing(monkeypatch):
    monkeypatch.setattr(coros_mcp, "_open", AsyncMock(side_effect=RuntimeError("COROS down")))

    with pytest.raises(RuntimeError):
        await coros_mcp.get_toolset("u1", "tok")

    assert "u1" not in coros_mcp._sessions


@pytest.mark.asyncio
async def test_discard_closes_users_session(_fake_open):
    toolset = (await coros_mcp.get_toolset("u1", "tok")).wrapped

    await coros_mcp.discard("u1")

    assert toolset.exited == 1
    assert "u1" not in coros_mcp._sessions


@pytest.mark.asyncio
async def test_pool_size_is_capped(_fake_open, monkeypatch):
    monkeypatch.setattr(coros_mcp, "MAX_SESSIONS", 2)
    for i in range(4):
        await coros_mcp.get_toolset(f"u{i}", "tok")
    await _settle()

    assert len(coros_mcp._sessions) == 2
    assert sum(t.exited for t in _FakeMCPToolset.instances) == 2


@pytest.mark.asyncio
async def test_close_all_closes_everything(_fake_open):
    await coros_mcp.get_toolset("u1", "tok")
    await coros_mcp.get_toolset("u2", "tok")

    await coros_mcp.close_all()

    assert coros_mcp._sessions == {}
    assert all(t.exited == 1 for t in _FakeMCPToolset.instances)
