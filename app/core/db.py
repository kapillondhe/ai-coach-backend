from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def asyncpg_url(database_url: str) -> str:
    _, _, rest = database_url.partition("://")
    return f"postgresql+asyncpg://{rest}"


def _get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Build the engine on first use.

    Synchronous on purpose: `create_async_engine` doesn't connect, so there is no
    await between the None-check and the assignment, and concurrent first requests
    can't each build (and leak) their own engine.
    """
    global _engine, _sessionmaker
    if _sessionmaker is None:
        _engine = create_async_engine(
            asyncpg_url(get_settings().database_url),
            connect_args={"statement_cache_size": 0},
        )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _sessionmaker


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with _get_sessionmaker()() as session:
        yield session


async def close_engine() -> None:
    global _engine, _sessionmaker
    engine = _engine
    _engine = None
    _sessionmaker = None
    if engine is not None:
        await engine.dispose()
