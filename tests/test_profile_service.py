from types import SimpleNamespace

import pytest

from app.services import profile as profile_service
from app.services.profile import ProfileError


class _FakeProfileSession:

    def __init__(self, store: dict[str, dict]):
        self.store = store

    async def execute(self, stmt):
        compiled_type = type(stmt).__name__
        if compiled_type == "Insert":
            values = stmt.compile().params
            user_id = values["user_id"]
            if user_id not in self.store:
                self.store[user_id] = {
                    "name": None,
                    "weight_kg": None,
                    "injury_notes": None,
                    "field_sources": dict(values.get("field_sources") or {}),
                }
            return SimpleNamespace(first=lambda: None)

        if compiled_type == "Update":
            values = stmt.compile().params
            user_id = values.pop("profiles_user_id", None)
            if user_id is None:
                # single-table update whens where clause param name varies;
                # fall back to matching the only stored user for this test helper
                (user_id,) = self.store.keys()
            row = self.store[user_id]
            for key, val in values.items():
                if key in row or key == "field_sources":
                    row[key] = val
            return SimpleNamespace(first=lambda: None)

        if compiled_type == "Select":
            # our select always filters on a single user_id; extract it from params
            params = stmt.compile().params
            (user_id,) = [v for v in params.values() if isinstance(v, str)]
            row = self.store.get(user_id)
            if row is None:
                return SimpleNamespace(first=lambda: None)
            return SimpleNamespace(first=lambda: SimpleNamespace(**row))

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
    monkeypatch.setattr(profile_service, "get_session", lambda: _FakeProfileSession(store))
    yield


@pytest.mark.asyncio
async def test_get_or_create_profile_creates_row_with_defaults(store):
    data = await profile_service.get_or_create_profile("user-1")

    assert data.user_id == "user-1"
    assert data.field_sources == {}
    assert data.name is None


@pytest.mark.asyncio
async def test_get_or_create_profile_idempotent(store):
    first = await profile_service.get_or_create_profile("user-1")
    store["user-1"]["name"] = "Kapil"  # simulate a real prior write
    second = await profile_service.get_or_create_profile("user-1")

    assert second.name == "Kapil"
    assert first.user_id == second.user_id


@pytest.mark.asyncio
async def test_update_profile_sets_fields_and_source(store):
    data = await profile_service.update_profile(
        "user-1", {"weight_kg": 72.5, "name": "Kapil"}, source="user"
    )

    assert data.weight_kg == 72.5
    assert data.name == "Kapil"
    assert data.field_sources == {"weight_kg": "user", "name": "user"}


@pytest.mark.asyncio
async def test_update_profile_rejects_unknown_field(store):
    with pytest.raises(ProfileError):
        await profile_service.update_profile("user-1", {"favorite_color": "blue"})


@pytest.mark.asyncio
async def test_update_profile_preserves_existing_field_sources(store):
    await profile_service.update_profile("user-1", {"name": "Kapil"}, source="chat")
    data = await profile_service.update_profile("user-1", {"weight_kg": 70}, source="user")

    assert data.field_sources == {"name": "chat", "weight_kg": "user"}


@pytest.mark.asyncio
async def test_update_profile_overwrites_source_on_re_edit(store):
    await profile_service.update_profile("user-1", {"name": "Kapil"}, source="chat")
    data = await profile_service.update_profile("user-1", {"name": "Kapil L"}, source="user")

    assert data.name == "Kapil L"
    assert data.field_sources["name"] == "user"
