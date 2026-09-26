
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, insert, select

from app.core.db import get_session
from app.core.models import UserMemory


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
    return MemoryData(id=memory_id, content=content, created_at=now)


async def list_memories(user_id: str) -> list[MemoryData]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(UserMemory.id, UserMemory.content, UserMemory.created_at)
                .where(UserMemory.user_id == user_id)
                .order_by(UserMemory.created_at)
            )
        ).all()
    return [MemoryData(id=r.id, content=r.content, created_at=r.created_at) for r in rows]


async def forget(user_id: str, memory_id: str) -> bool:
    async with get_session() as session:
        result = await session.execute(
            delete(UserMemory).where(UserMemory.id == memory_id, UserMemory.user_id == user_id)
        )
        await session.commit()
    return result.rowcount > 0
