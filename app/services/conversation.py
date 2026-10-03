
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, insert, select, update

from app.core.db import get_session
from app.core.models import Conversation, ConversationMessage

ROLES = ("user", "assistant")


class ConversationError(ValueError):
    pass


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation_id doesn't exist, or doesn't belong to the given user_id."""


@dataclass
class MessageData:
    role: str
    content: str


@dataclass
class ConversationData:
    id: str
    user_id: str
    title: str | None
    messages: list[MessageData] = field(default_factory=list)


@dataclass
class ConversationSummary:
    id: str
    title: str | None
    updated_at: datetime


async def create_conversation(user_id: str, *, title: str | None = None) -> str:
    conversation_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    async with get_session() as session:
        await session.execute(
            insert(Conversation).values(
                id=conversation_id,
                user_id=user_id,
                title=title,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    return conversation_id


async def get_conversation(conversation_id: str, user_id: str) -> ConversationData:
    """Fetch a conversation and its messages, scoped to user_id (ownership check included).

    One round trip: an outer join, so a conversation with no messages yet still
    returns its row (with NULL role/content).
    """
    async with get_session() as session:
        rows = (
            await session.execute(
                select(
                    Conversation.id,
                    Conversation.user_id,
                    Conversation.title,
                    ConversationMessage.role,
                    ConversationMessage.content,
                )
                .select_from(Conversation)
                .outerjoin(ConversationMessage, ConversationMessage.conversation_id == Conversation.id)
                .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
                .order_by(ConversationMessage.created_at)
            )
        ).all()
    if not rows:
        raise ConversationNotFoundError(conversation_id)

    first = rows[0]
    return ConversationData(
        id=first.id,
        user_id=first.user_id,
        title=first.title,
        messages=[MessageData(role=r.role, content=r.content) for r in rows if r.role is not None],
    )


async def append_messages(conversation_id: str, user_id: str, messages: list[MessageData]) -> None:
    """Append messages to a user's conversation.

    The ownership check is the `updated_at` bump itself (scoped by user_id), and
    all messages go in one multi-row insert — 3 round trips total incl. commit.
    """
    for msg in messages:
        if msg.role not in ROLES:
            raise ConversationError(f"Unknown message role: {msg.role!r}")

    now = datetime.now(UTC)
    async with get_session() as session:
        result = await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .values(updated_at=now)
        )
        if result.rowcount == 0:
            await session.rollback()
            raise ConversationNotFoundError(conversation_id)
        if messages:
            await session.execute(
                insert(ConversationMessage).values(
                    [
                        {
                            "id": str(uuid.uuid4()),
                            "conversation_id": conversation_id,
                            "role": msg.role,
                            "content": msg.content,
                            # Distinct timestamps keep a turn's user/assistant order stable,
                            # since history is read back ordered by created_at.
                            "created_at": now + timedelta(microseconds=i),
                        }
                        for i, msg in enumerate(messages)
                    ]
                )
            )
        await session.commit()


async def set_title(conversation_id: str, user_id: str, title: str) -> None:
    async with get_session() as session:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .values(title=title)
        )
        await session.commit()


async def delete_conversation(conversation_id: str, user_id: str) -> bool:
    async with get_session() as session:
        result = await session.execute(
            delete(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        )
        deleted = result.rowcount > 0
        if deleted:
            await session.execute(
                delete(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id)
            )
        await session.commit()
    return deleted


async def list_conversations(user_id: str) -> list[ConversationSummary]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(Conversation.id, Conversation.title, Conversation.updated_at)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
            )
        ).all()
    return [ConversationSummary(id=r.id, title=r.title, updated_at=r.updated_at) for r in rows]
