import uuid
from types import SimpleNamespace

import pytest

from app.services import conversation as conversation_service
from app.services.conversation import ConversationNotFoundError, MessageData
from tests.fakes import FakeSession, multirow_values, stmt_kind, stmt_params


class _FakeConversationSession(FakeSession):
    """In-memory fake standing in for get_session(), mirroring _FakeProfileSession's shape."""

    def __init__(self, conversations: dict, messages: dict):
        super().__init__()
        self.conversations = conversations
        self.messages = messages
        self.insert_statements = 0
        self.select_statements = 0

    async def handle(self, stmt):
        compiled_type = stmt_kind(stmt)

        if compiled_type == "Insert":
            table = stmt.table.name
            for row in multirow_values(stmt):
                if table == "conversations":
                    self.conversations[row["id"]] = row
                elif table == "conversation_messages":
                    self.messages.setdefault(row["conversation_id"], []).append(row)
            self.insert_statements += 1
            return SimpleNamespace(first=lambda: None)

        if compiled_type == "Update":
            params = stmt_params(stmt)
            conv_id = params.get("id_1")
            owner = params.get("user_id_1")
            row = self.conversations.get(conv_id)
            if row is None or (owner is not None and row["user_id"] != owner):
                return SimpleNamespace(rowcount=0, first=lambda: None)
            for key, val in params.items():
                if key in row:
                    row[key] = val
            return SimpleNamespace(rowcount=1, first=lambda: None)

        if compiled_type == "Delete":
            table = stmt.table.name
            params = stmt_params(stmt)
            str_params = [v for v in params.values() if isinstance(v, str)]

            if table == "conversations":
                matches = [c for c in self.conversations.values() if all(v in c.values() for v in str_params)]
                for c in matches:
                    del self.conversations[c["id"]]
                    # Mirrors conversation_messages' FK ON DELETE CASCADE in Postgres.
                    self.messages.pop(c["id"], None)
                return SimpleNamespace(rowcount=len(matches))

            raise AssertionError(f"Unhandled delete table: {table}")

        if compiled_type == "Select":
            from_ = stmt.get_final_froms()[0]
            params = stmt_params(stmt)
            str_params = [v for v in params.values() if isinstance(v, str)]

            if from_.__class__.__name__.endswith("Join"):
                # get_conversation: conversations OUTER JOIN conversation_messages.
                self.select_statements += 1
                conv = self.conversations.get(params["id_1"])
                if conv is None or conv["user_id"] != params["user_id_1"]:
                    return SimpleNamespace(all=lambda: [])
                msgs = sorted(self.messages.get(conv["id"], []), key=lambda m: m.get("created_at") or 0)
                base = {"id": conv["id"], "user_id": conv["user_id"], "title": conv.get("title")}
                if not msgs:
                    joined = [SimpleNamespace(**base, role=None, content=None)]
                else:
                    joined = [SimpleNamespace(**base, role=m["role"], content=m["content"]) for m in msgs]
                return SimpleNamespace(all=lambda: joined)

            table_name = from_.name

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

        return await super().handle(stmt)


@pytest.fixture
def store():
    return {"conversations": {}, "messages": {}, "sessions": []}


@pytest.fixture(autouse=True)
def _patch_session(monkeypatch, store):
    def _new_session():
        session = _FakeConversationSession(store["conversations"], store["messages"])
        store["sessions"].append(session)
        return session

    monkeypatch.setattr(conversation_service, "get_session", _new_session)
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
        await conversation_service.append_messages(conversation_id, "user-2", [MessageData(role="user", content="hi")])

    assert store["messages"] == {}


@pytest.mark.asyncio
async def test_append_messages_raises_for_unknown_conversation(store):
    with pytest.raises(ConversationNotFoundError):
        await conversation_service.append_messages(
            str(uuid.uuid4()), "user-1", [MessageData(role="user", content="hi")]
        )

    assert store["messages"] == {}


@pytest.mark.asyncio
async def test_append_messages_writes_a_turn_in_one_insert_and_keeps_order(store):
    conversation_id = await conversation_service.create_conversation("user-1")
    store["sessions"].clear()

    await conversation_service.append_messages(
        conversation_id,
        "user-1",
        [MessageData(role="user", content="q"), MessageData(role="assistant", content="a")],
    )

    (session,) = store["sessions"]  # one session for the whole write
    assert session.insert_statements == 1  # both messages in a single multi-row insert
    rows = store["messages"][conversation_id]
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["created_at"] < rows[1]["created_at"]  # stable order on read-back
    assert store["conversations"][conversation_id]["updated_at"] == rows[0]["created_at"]


@pytest.mark.asyncio
async def test_get_conversation_uses_a_single_query(store):
    conversation_id = await conversation_service.create_conversation("user-1")
    await conversation_service.append_messages(conversation_id, "user-1", [MessageData(role="user", content="hi")])
    store["sessions"].clear()

    await conversation_service.get_conversation(conversation_id, "user-1")

    (session,) = store["sessions"]
    assert session.select_statements == 1


@pytest.mark.asyncio
async def test_get_conversation_with_no_messages_returns_empty_list(store):
    conversation_id = await conversation_service.create_conversation("user-1")

    data = await conversation_service.get_conversation(conversation_id, "user-1")

    assert data.id == conversation_id
    assert data.messages == []


@pytest.mark.asyncio
async def test_list_conversations_scoped_to_user(store):
    conv_a = await conversation_service.create_conversation("user-1")
    await conversation_service.create_conversation("user-2")

    summaries = await conversation_service.list_conversations("user-1")

    assert [s.id for s in summaries] == [conv_a]


@pytest.mark.asyncio
async def test_delete_conversation_removes_conversation_and_messages(store):
    conversation_id = await conversation_service.create_conversation("user-1")
    await conversation_service.append_messages(conversation_id, "user-1", [MessageData(role="user", content="hi")])

    deleted = await conversation_service.delete_conversation(conversation_id, "user-1")

    assert deleted is True
    assert conversation_id not in store["conversations"]
    assert conversation_id not in store["messages"]
    with pytest.raises(ConversationNotFoundError):
        await conversation_service.get_conversation(conversation_id, "user-1")


@pytest.mark.asyncio
async def test_delete_conversation_returns_false_for_wrong_user(store):
    conversation_id = await conversation_service.create_conversation("user-1")

    deleted = await conversation_service.delete_conversation(conversation_id, "user-2")

    assert deleted is False
    assert conversation_id in store["conversations"]


@pytest.mark.asyncio
async def test_delete_conversation_returns_false_for_unknown_id(store):
    deleted = await conversation_service.delete_conversation(str(uuid.uuid4()), "user-1")

    assert deleted is False
