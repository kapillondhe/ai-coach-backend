import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

logger = logging.getLogger(__name__)

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


async def warm_up() -> None:
    """Open one pooled connection at startup so the first request skips the ~1s connect.

    Best-effort: a DB that's unreachable at boot must not stop the app from starting.
    """
    try:
        async with get_session() as session:
            await session.execute(text("select 1"))
    except Exception:
        logger.warning("Database warm-up failed; first request will connect lazily", exc_info=True)


async def close_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
