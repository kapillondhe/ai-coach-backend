from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def asyncpg_url(database_url: str) -> str:
    _, _, rest = database_url.partition("://")
    return f"postgresql+asyncpg://{rest}"


async def _init_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _engine
    settings = get_settings()

    _engine = create_async_engine(
        asyncpg_url(settings.database_url),
        connect_args={"statement_cache_size": 0},
    )
    return async_sessionmaker(_engine, expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = await _init_sessionmaker()
    async with _sessionmaker() as session:
        yield session


async def close_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
