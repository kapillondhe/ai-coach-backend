
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, insert, select

from app.core.db import get_session
from app.core.models import UserMemory

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


async def remember(user_id: str, content: str) -> MemoryData:
    memory_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    async with get_session() as session:
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
        result = await session.execute(
            delete(UserMemory).where(UserMemory.id == memory_id, UserMemory.user_id == user_id)
        )
        await session.commit()
    _list_cache.pop(user_id, None)
    return result.rowcount > 0
