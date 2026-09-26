from types import SimpleNamespace

import pytest

from app.services import memory as memory_service


class _FakeMemorySession:
    def __init__(self, store: dict[str, list[dict]]):
        self.store = store

    async def execute(self, stmt):
        compiled_type = type(stmt).__name__

        if compiled_type == "Insert":
            values = stmt.compile().params
            self.store.setdefault(values["user_id"], []).append(dict(values))
            return SimpleNamespace(first=lambda: None)

        if compiled_type == "Select":
            params = stmt.compile().params
            (user_id,) = [v for v in params.values() if isinstance(v, str)]
            rows = self.store.get(user_id, [])
            ordered = sorted(rows, key=lambda m: m["created_at"])
            return SimpleNamespace(
                all=lambda: [SimpleNamespace(id=m["id"], content=m["content"], created_at=m["created_at"]) for m in ordered]
            )

        if compiled_type == "Delete":
            params = stmt.compile().params
            memory_id = params["id_1"]
            user_id = params["user_id_1"]
            rows = self.store.get(user_id, [])
            before = len(rows)
            self.store[user_id] = [m for m in rows if m["id"] != memory_id]
            deleted = before - len(self.store[user_id])
            return SimpleNamespace(rowcount=deleted)

        raise AssertionError(f"Unexpected statement type: {compiled_type}")

    async def commit(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def store():
    return {}


@pytest.fixture(autouse=True)
def _patch_session(monkeypatch, store):
    monkeypatch.setattr(memory_service, "get_session", lambda: _FakeMemorySession(store))
    yield


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
