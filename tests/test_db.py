"""app.core.db: lazy engine init is race-free and close_engine resets it.

No database is touched: `create_async_engine` doesn't connect, and opening an
AsyncSession doesn't either until a statement runs.
"""

import asyncio

import pytest

from app.core import db


@pytest.fixture(autouse=True)
async def _reset_engine():
    await db.close_engine()
    yield
    await db.close_engine()


async def test_concurrent_first_sessions_build_one_engine(monkeypatch):
    built = []
    real_create = db.create_async_engine

    def _counting_create(*args, **kwargs):
        built.append(args)
        return real_create(*args, **kwargs)

    monkeypatch.setattr(db, "create_async_engine", _counting_create)

    async def _open_session():
        async with db.get_session() as session:
            await asyncio.sleep(0)
            return session

    sessions = await asyncio.gather(*(_open_session() for _ in range(10)))

    assert len(built) == 1
    assert len({id(s) for s in sessions}) == 10  # still one session per caller
    assert built[0][0].startswith("postgresql+asyncpg://")


async def test_close_engine_disposes_and_allows_reinit(monkeypatch):
    first = db._get_sessionmaker()
    engine = db._engine
    assert engine is not None

    await db.close_engine()

    assert db._engine is None
    assert db._sessionmaker is None
    second = db._get_sessionmaker()
    assert second is not first
    assert db._engine is not engine


async def test_close_engine_without_engine_is_a_noop():
    await db.close_engine()
    await db.close_engine()
    assert db._engine is None
