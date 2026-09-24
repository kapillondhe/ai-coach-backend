import uuid
from types import SimpleNamespace

import pytest

from app.services import conversation as conversation_service
from app.services.conversation import ConversationNotFoundError, MessageData


class _FakeConversationSession:
    """In-memory fake standing in for get_session(), mirroring _FakeProfileSession's shape."""

    def __init__(self, conversations: dict, messages: dict):
        self.conversations = conversations
        self.messages = messages

    async def execute(self, stmt):
        compiled_type = type(stmt).__name__

        if compiled_type == "Insert":
            values = stmt.compile().params
            table = stmt.table.name
            if table == "conversations":
                self.conversations[values["id"]] = dict(values)
            elif table == "conversation_messages":
                self.messages.setdefault(values["conversation_id"], []).append(dict(values))
            return SimpleNamespace(first=lambda: None)

        if compiled_type == "Update":
            values = stmt.compile().params
            conv_id = values.get("conversations_id") or next(iter(self.conversations))
            row = self.conversations[conv_id]
            for key, val in values.items():
                if key in row:
                    row[key] = val
            return SimpleNamespace(first=lambda: None)

        if compiled_type == "Select":
            table_name = stmt.get_final_froms()[0].name
            params = stmt.compile().params
            str_params = [v for v in params.values() if isinstance(v, str)]

            if table_name == "conversations":
                matches = [c for c in self.conversations.values() if all(v in c.values() for v in str_params)]
                if not matches:
                    return SimpleNamespace(first=lambda: None, all=lambda: [])
                row = matches[0]
                ordered = sorted(matches, key=lambda c: c.get("updated_at") or 0, reverse=True)
                return SimpleNamespace(
                    first=lambda: SimpleNamespace(id=row["id"], user_id=row["user_id"], title=row.get("title")),
                    all=lambda: [
                        SimpleNamespace(id=c["id"], title=c.get("title"), updated_at=c.get("updated_at"))
                        for c in ordered
                    ],
                )

            if table_name == "conversation_messages":
                (conversation_id,) = str_params
                rows = self.messages.get(conversation_id, [])
                ordered = sorted(rows, key=lambda m: m.get("created_at") or 0)
                return SimpleNamespace(
                    all=lambda: [SimpleNamespace(role=m["role"], content=m["content"]) for m in ordered]
                )

            raise AssertionError(f"Unhandled select table: {table_name}")

        raise AssertionError(f"Unexpected statement type: {compiled_type}")

    async def commit(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def store():
    return {"conversations": {}, "messages": {}}


@pytest.fixture(autouse=True)
def _patch_session(monkeypatch, store):
    monkeypatch.setattr(
        conversation_service,
        "get_session",
        lambda: _FakeConversationSession(store["conversations"], store["messages"]),
    )
    yield


@pytest.mark.asyncio
async def test_create_conversation_returns_new_id(store):
    conversation_id = await conversation_service.create_conversation("user-1")

    assert conversation_id in store["conversations"]
    assert store["conversations"][conversation_id]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_append_messages_and_get_conversation_round_trips(store):
    conversation_id = await conversation_service.create_conversation("user-1")

    await conversation_service.append_messages(
        conversation_id,
        "user-1",
        [MessageData(role="user", content="hello"), MessageData(role="assistant", content="hi there")],
    )

    data = await conversation_service.get_conversation(conversation_id, "user-1")

    assert data.id == conversation_id
    assert [m.role for m in data.messages] == ["user", "assistant"]
    assert [m.content for m in data.messages] == ["hello", "hi there"]


@pytest.mark.asyncio
async def test_get_conversation_raises_for_wrong_user(store):
    conversation_id = await conversation_service.create_conversation("user-1")

    with pytest.raises(ConversationNotFoundError):
        await conversation_service.get_conversation(conversation_id, "user-2")


@pytest.mark.asyncio
async def test_get_conversation_raises_for_unknown_id(store):
    with pytest.raises(ConversationNotFoundError):
        await conversation_service.get_conversation(str(uuid.uuid4()), "user-1")


@pytest.mark.asyncio
async def test_append_messages_raises_for_wrong_user(store):
    conversation_id = await conversation_service.create_conversation("user-1")

    with pytest.raises(ConversationNotFoundError):
        await conversation_service.append_messages(
            conversation_id, "user-2", [MessageData(role="user", content="hi")]
        )


@pytest.mark.asyncio
async def test_list_conversations_scoped_to_user(store):
    conv_a = await conversation_service.create_conversation("user-1")
    await conversation_service.create_conversation("user-2")

    summaries = await conversation_service.list_conversations("user-1")

    assert [s.id for s in summaries] == [conv_a]
