"""Long-term per-user memory: short facts the coach saved via its `remember` tool."""

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import CursorResult, delete, func, insert, select

from app.core.db import get_session
from app.core.models import UserMemory

# Memory bounds. Every stored fact can end up in every system prompt, so both
# fact size and count are capped to keep prompt size (and cost) bounded.
MAX_MEMORY_LENGTH = 300  # chars per fact; enforced by the `remember` tool
MAX_STORED_MEMORIES = 100  # per user; enforced by `remember()` below
MAX_PROMPT_MEMORIES = 50  # newest N injected into the system prompt

# Per-process cache of list_memories results, read on every signed-in chat turn.
# Writes in this process invalidate immediately; another worker process may serve
# a stale list for up to the TTL.
_LIST_CACHE_TTL = 60.0
_list_cache: dict[str, tuple[float, list["MemoryData"]]] = {}


@dataclass
class MemoryData:
    id: str
    content: str
    created_at: datetime


class MemoryFullError(Exception):
    """The user already has `MAX_STORED_MEMORIES` saved facts."""


async def remember(user_id: str, content: str) -> MemoryData:
    """Store a fact; raises `MemoryFullError` once the user is at the cap.

    Count-then-insert isn't atomic, so concurrent saves can overshoot the cap by
    a few rows; that's fine for a soft bound on prompt size.
    """
    memory_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    async with get_session() as session:
        count = (
            await session.execute(select(func.count()).select_from(UserMemory).where(UserMemory.user_id == user_id))
        ).scalar_one()
        if count >= MAX_STORED_MEMORIES:
            raise MemoryFullError(user_id)
        await session.execute(
            insert(UserMemory).values(
                id=memory_id,
                user_id=user_id,
                content=content,
                created_at=now,
            )
        )
        await session.commit()
    _list_cache.pop(user_id, None)
    return MemoryData(id=memory_id, content=content, created_at=now)


async def list_memories(user_id: str) -> list[MemoryData]:
    cached = _list_cache.get(user_id)
    if cached is not None and time.monotonic() - cached[0] < _LIST_CACHE_TTL:
        return list(cached[1])

    async with get_session() as session:
        rows = (
            await session.execute(
                select(UserMemory.id, UserMemory.content, UserMemory.created_at)
                .where(UserMemory.user_id == user_id)
                .order_by(UserMemory.created_at)
            )
        ).all()
    memories = [MemoryData(id=r.id, content=r.content, created_at=r.created_at) for r in rows]
    _list_cache[user_id] = (time.monotonic(), memories)
    return list(memories)


async def forget(user_id: str, memory_id: str) -> bool:
    async with get_session() as session:
        result = cast(
            CursorResult,
            await session.execute(delete(UserMemory).where(UserMemory.id == memory_id, UserMemory.user_id == user_id)),
        )
        await session.commit()
    _list_cache.pop(user_id, None)
    return result.rowcount > 0
