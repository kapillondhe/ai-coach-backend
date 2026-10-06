from types import SimpleNamespace

import pytest

from app.services import memory as memory_service
from tests.fakes import FakeSession, stmt_kind, stmt_params


class _FakeMemorySession(FakeSession):
    def __init__(self, store: dict[str, list[dict]]):
        super().__init__()
        self.store = store

    async def handle(self, stmt):
        compiled_type = stmt_kind(stmt)
        params = stmt_params(stmt)

        if compiled_type == "Insert":
            self.store.setdefault(params["user_id"], []).append(params)
            return SimpleNamespace(first=lambda: None)

        if compiled_type == "Select" and "count(" in str(stmt).lower():
            (user_id,) = params.values()
            count = len(self.store.get(user_id, []))
            return SimpleNamespace(scalar_one=lambda: count)

        if compiled_type == "Select":
            (user_id,) = [v for v in params.values() if isinstance(v, str)]
            rows = self.store.get(user_id, [])
            ordered = sorted(rows, key=lambda m: m["created_at"])
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(id=m["id"], content=m["content"], created_at=m["created_at"]) for m in ordered
                ]
            )

        if compiled_type == "Delete":
            memory_id = params["id_1"]
            user_id = params["user_id_1"]
            rows = self.store.get(user_id, [])
            before = len(rows)
            self.store[user_id] = [m for m in rows if m["id"] != memory_id]
            deleted = before - len(self.store[user_id])
            return SimpleNamespace(rowcount=deleted)

        return await super().handle(stmt)


@pytest.fixture
def store():
    return {}


@pytest.fixture(autouse=True)
def _patch_session(monkeypatch, store):
    monkeypatch.setattr(memory_service, "get_session", lambda: _FakeMemorySession(store))
    memory_service._list_cache.clear()
    yield
    memory_service._list_cache.clear()


@pytest.mark.asyncio
async def test_list_memories_is_cached_between_calls(store, monkeypatch):
    await memory_service.remember("user-1", "first fact")
    await memory_service.list_memories("user-1")

    store["user-1"][0]["content"] = "edited elsewhere"
    memories = await memory_service.list_memories("user-1")

    assert [m.content for m in memories] == ["first fact"]  # served from cache, no DB read


@pytest.mark.asyncio
async def test_list_memories_cache_expires_after_ttl(store, monkeypatch):
    await memory_service.remember("user-1", "first fact")
    await memory_service.list_memories("user-1")
    monkeypatch.setattr(memory_service, "_LIST_CACHE_TTL", 0.0)

    await memory_service.remember("user-2", "unrelated")  # doesn't touch user-1's entry
    store["user-1"][0]["content"] = "edited elsewhere"
    memories = await memory_service.list_memories("user-1")

    assert [m.content for m in memories] == ["edited elsewhere"]


@pytest.mark.asyncio
async def test_remember_invalidates_cached_list(store):
    await memory_service.remember("user-1", "first fact")
    await memory_service.list_memories("user-1")

    await memory_service.remember("user-1", "second fact")
    memories = await memory_service.list_memories("user-1")

    assert [m.content for m in memories] == ["first fact", "second fact"]


@pytest.mark.asyncio
async def test_forget_invalidates_cached_list(store):
    data = await memory_service.remember("user-1", "first fact")
    await memory_service.list_memories("user-1")

    await memory_service.forget("user-1", data.id)

    assert await memory_service.list_memories("user-1") == []


@pytest.mark.asyncio
async def test_list_memories_cache_is_not_mutated_by_callers(store):
    await memory_service.remember("user-1", "first fact")
    first = await memory_service.list_memories("user-1")
    first.clear()

    assert [m.content for m in await memory_service.list_memories("user-1")] == ["first fact"]


@pytest.mark.asyncio
async def test_remember_stores_a_fact(store):
    data = await memory_service.remember("user-1", "Training for a first 70.3 in June")

    assert data.content == "Training for a first 70.3 in June"
    assert store["user-1"][0]["content"] == data.content


@pytest.mark.asyncio
async def test_list_memories_scoped_to_user_and_ordered(store):
    await memory_service.remember("user-1", "first fact")
    await memory_service.remember("user-1", "second fact")
    await memory_service.remember("user-2", "other user's fact")

    memories = await memory_service.list_memories("user-1")

    assert [m.content for m in memories] == ["first fact", "second fact"]


@pytest.mark.asyncio
async def test_list_memories_empty_for_unknown_user(store):
    memories = await memory_service.list_memories("user-nobody")

    assert memories == []


@pytest.mark.asyncio
async def test_forget_deletes_owned_memory(store):
    data = await memory_service.remember("user-1", "first fact")

    deleted = await memory_service.forget("user-1", data.id)

    assert deleted is True
    assert await memory_service.list_memories("user-1") == []


@pytest.mark.asyncio
async def test_forget_returns_false_for_unknown_memory(store):
    deleted = await memory_service.forget("user-1", "no-such-id")

    assert deleted is False


@pytest.mark.asyncio
async def test_forget_does_not_delete_another_users_memory(store):
    data = await memory_service.remember("user-1", "first fact")

    deleted = await memory_service.forget("user-2", data.id)

    assert deleted is False
    memories = await memory_service.list_memories("user-1")
    assert [m.content for m in memories] == ["first fact"]


@pytest.mark.asyncio
async def test_remember_raises_memory_full_at_stored_cap(store, monkeypatch):
    monkeypatch.setattr(memory_service, "MAX_STORED_MEMORIES", 2)
    await memory_service.remember("user-1", "first fact")
    await memory_service.remember("user-1", "second fact")

    with pytest.raises(memory_service.MemoryFullError):
        await memory_service.remember("user-1", "third fact")

    assert [m["content"] for m in store["user-1"]] == ["first fact", "second fact"]


@pytest.mark.asyncio
async def test_memory_cap_is_per_user(store, monkeypatch):
    monkeypatch.setattr(memory_service, "MAX_STORED_MEMORIES", 1)
    await memory_service.remember("user-1", "first fact")

    data = await memory_service.remember("user-2", "other user's fact")

    assert data.content == "other user's fact"


def test_memory_bounds_are_consistent():
    assert 0 < memory_service.MAX_PROMPT_MEMORIES <= memory_service.MAX_STORED_MEMORIES
    assert memory_service.MAX_MEMORY_LENGTH > 0
