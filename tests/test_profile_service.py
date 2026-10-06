from types import SimpleNamespace

import pytest

from app.services import profile as profile_service
from app.services.profile import ProfileError
from tests.fakes import FakeSession, stmt_kind, stmt_params


class _Result:
    def __init__(self, row: dict | None) -> None:
        self._row = row

    def one(self):
        if self._row is None:
            raise AssertionError("statement returned no row")
        return SimpleNamespace(**self._row)


class _FakeProfileSession(FakeSession):
    """Emulates the two statements the service issues, against an in-memory row store.

    Insert = INSERT ... ON CONFLICT (user_id) DO UPDATE ... RETURNING (get-or-create).
    Update = UPDATE ... WHERE user_id = ? RETURNING. Anything else (e.g. a follow-up
    SELECT) fails the test, which pins the round-trip count.
    """

    def __init__(self, store: dict[str, dict], log: list[str]) -> None:
        super().__init__()
        self.store = store
        self.log = log

    async def handle(self, stmt):
        kind = stmt_kind(stmt)
        self.log.append(kind)
        params = stmt_params(stmt)

        if kind == "Insert":
            assert stmt._post_values_clause is not None, "get-or-create must be an upsert"
            row = self.store.setdefault(
                params["user_id"],
                {"name": None, "weight_kg": None, "injury_notes": None, "field_sources": dict(params["field_sources"])},
            )
            return _Result(dict(row))

        if kind == "Update":
            row = self.store.get(params.pop("user_id_1"))
            if row is None:
                return _Result(None)
            params.pop("updated_at")
            row.update(params)
            return _Result(dict(row))

        return await super().handle(stmt)

    async def commit(self):
        await super().commit()
        self.log.append("commit")

    async def __aenter__(self):
        self.log.append("begin")
        return self


@pytest.fixture
def store():
    return {}


@pytest.fixture
def log():
    return []


@pytest.fixture(autouse=True)
def _patch_session(monkeypatch, store, log):
    monkeypatch.setattr(profile_service, "get_session", lambda: _FakeProfileSession(store, log))


@pytest.mark.asyncio
async def test_get_or_create_profile_creates_row_with_defaults(store, log):
    data = await profile_service.get_or_create_profile("user-1")

    assert data.user_id == "user-1"
    assert data.field_sources == {}
    assert data.name is None
    assert "user-1" in store
    assert log == ["begin", "Insert", "commit"]  # one round trip


@pytest.mark.asyncio
async def test_get_or_create_profile_idempotent(store):
    first = await profile_service.get_or_create_profile("user-1")
    store["user-1"]["name"] = "Kapil"  # simulate a real prior write
    second = await profile_service.get_or_create_profile("user-1")

    assert second.name == "Kapil"
    assert first.user_id == second.user_id
    assert list(store) == ["user-1"]


@pytest.mark.asyncio
async def test_update_profile_sets_fields_and_source(store, log):
    data = await profile_service.update_profile("user-1", {"weight_kg": 72.5, "name": "Kapil"}, source="user")

    assert data.weight_kg == 72.5
    assert data.name == "Kapil"
    assert data.field_sources == {"weight_kg": "user", "name": "user"}
    assert store["user-1"]["field_sources"] == {"weight_kg": "user", "name": "user"}
    # Two statements in a single transaction; the response comes from RETURNING.
    assert log == ["begin", "Insert", "Update", "commit"]


@pytest.mark.asyncio
async def test_update_profile_rejects_unknown_field(store, log):
    with pytest.raises(ProfileError):
        await profile_service.update_profile("user-1", {"favorite_color": "blue"})

    assert log == []  # validated before touching the DB


@pytest.mark.asyncio
async def test_update_profile_rejects_field_sources_as_a_field(store):
    with pytest.raises(ProfileError):
        await profile_service.update_profile("user-1", {"field_sources": {"name": "chat"}})


@pytest.mark.asyncio
async def test_update_profile_preserves_existing_field_sources(store):
    await profile_service.update_profile("user-1", {"name": "Kapil"}, source="chat")
    data = await profile_service.update_profile("user-1", {"weight_kg": 70}, source="user")

    assert data.field_sources == {"name": "chat", "weight_kg": "user"}
    assert data.name == "Kapil"


@pytest.mark.asyncio
async def test_update_profile_overwrites_source_on_re_edit(store):
    await profile_service.update_profile("user-1", {"name": "Kapil"}, source="chat")
    data = await profile_service.update_profile("user-1", {"name": "Kapil L"}, source="user")

    assert data.name == "Kapil L"
    assert data.field_sources["name"] == "user"


@pytest.mark.asyncio
async def test_update_profile_can_clear_a_field(store):
    await profile_service.update_profile("user-1", {"injury_notes": "sore knee"}, source="chat")
    data = await profile_service.update_profile("user-1", {"injury_notes": None}, source="user")

    assert data.injury_notes is None
    assert data.field_sources == {"injury_notes": "user"}
