"""Per-user pool of open COROS MCP sessions, reused across chat turns.

Opening a COROS MCP session (connect + initialize, then listing its ~34 tools) costs
~3-4s, versus ~0.3s per tool call on an already-open session. Building a fresh
`MCPToolset` per chat turn paid that on every message, before the model could start.
Here each user's session stays open and its tool list stays cached between turns,
until it's idle for `IDLE_TTL`, its access token changes, or a call on it fails.
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# A session idle for 5 minutes was verified to still work against COROS's server.
IDLE_TTL = 300.0
MAX_SESSIONS = 100
CONNECT_TIMEOUT = 10.0


@dataclass
class _Entry:
    token: str
    toolset: MCPToolset
    last_used: float


_sessions: dict[str, _Entry] = {}
_locks: dict[str, asyncio.Lock] = {}
# Strong refs to in-flight background closes so they aren't garbage-collected.
_closing: set[asyncio.Task[None]] = set()


@dataclass
class _EvictOnError(WrapperToolset[Any]):
    """Drops the user's pooled session when a call on it fails, so the next turn reconnects.

    `ModelRetry` is a tool-level error the model is meant to see and recover from, not a
    broken session, so it doesn't evict.
    """

    user_id: str = ""

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        try:
            return await super().get_tools(ctx)
        except ModelRetry:
            raise
        except Exception:
            _evict(self.user_id, self.wrapped)
            raise

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> Any:
        try:
            return await super().call_tool(name, tool_args, ctx, tool)
        except ModelRetry:
            raise
        except Exception:
            _evict(self.user_id, self.wrapped)
            raise


async def _open(access_token: str) -> MCPToolset:
    toolset = MCPToolset(
        client=get_settings().coros_mcp_server_url,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    await toolset.__aenter__()
    try:
        await toolset.list_tools()  # cached on the toolset for later turns
    except BaseException:
        with contextlib.suppress(Exception):
            await toolset.__aexit__(None, None, None)
        raise
    return toolset


async def _close(toolset: MCPToolset) -> None:
    # Only releases the pool's hold: a chat run still using this session keeps it open
    # until that run finishes (MCPToolset is reference-counted).
    try:
        await toolset.__aexit__(None, None, None)
    except Exception:
        logger.warning("Error closing a COROS MCP session", exc_info=True)


def _close_in_background(toolset: MCPToolset) -> None:
    task = asyncio.create_task(_close(toolset))
    _closing.add(task)
    task.add_done_callback(_closing.discard)


def _evict(user_id: str, toolset: AbstractToolset[Any]) -> None:
    entry = _sessions.get(user_id)
    if entry is not None and entry.toolset is toolset:
        del _sessions[user_id]
        _close_in_background(entry.toolset)


def _sweep(now: float) -> None:
    for user_id, entry in list(_sessions.items()):
        if now - entry.last_used > IDLE_TTL:
            del _sessions[user_id]
            _close_in_background(entry.toolset)
    while len(_sessions) > MAX_SESSIONS:
        oldest = min(_sessions, key=lambda uid: _sessions[uid].last_used)
        _close_in_background(_sessions.pop(oldest).toolset)


async def get_toolset(user_id: str, access_token: str) -> AbstractToolset[Any]:
    """Return this user's COROS toolset, opening a session only if there's no usable one.

    Raises if a new session can't be opened (COROS down, token rejected, timeout).
    """
    lock = _locks.setdefault(user_id, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        entry = _sessions.get(user_id)
        if entry is not None and (entry.token != access_token or now - entry.last_used > IDLE_TTL):
            del _sessions[user_id]
            _close_in_background(entry.toolset)
            entry = None
        if entry is None:
            toolset = await asyncio.wait_for(_open(access_token), CONNECT_TIMEOUT)
            entry = _Entry(token=access_token, toolset=toolset, last_used=now)
            _sessions[user_id] = entry
        entry.last_used = now
        _sweep(now)
    return _EvictOnError(entry.toolset, user_id=user_id)


async def discard(user_id: str) -> None:
    """Close a user's pooled session, e.g. after they disconnect COROS."""
    entry = _sessions.pop(user_id, None)
    if entry is not None:
        await _close(entry.toolset)


async def close_all() -> None:
    """Close every pooled session (app shutdown)."""
    entries = list(_sessions.values())
    _sessions.clear()
    await asyncio.gather(*(_close(e.toolset) for e in entries))
    if _closing:
        await asyncio.gather(*_closing, return_exceptions=True)
