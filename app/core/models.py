
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class _UTCDateTime(TypeDecorator):
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    pass


class OAuthClient(Base):
    __tablename__ = "oauth_clients"

    provider: Mapped[str] = mapped_column(primary_key=True)
    client_id: Mapped[str]
    client_secret: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime)


class OAuthState(Base):
    __tablename__ = "oauth_states"

    state: Mapped[str] = mapped_column(primary_key=True)
    provider: Mapped[str]
    user_id: Mapped[str]
    code_verifier: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime)
    expires_at: Mapped[datetime] = mapped_column(_UTCDateTime)


class UserIntegration(Base):
    __tablename__ = "user_integrations"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(primary_key=True)
    access_token_encrypted: Mapped[str]
    refresh_token_encrypted: Mapped[str | None]
    scope: Mapped[str | None]
    expires_at: Mapped[datetime | None] = mapped_column(_UTCDateTime)
    connected_at: Mapped[datetime] = mapped_column(_UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(_UTCDateTime)


class Profile(Base):
    __tablename__ = "profiles"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str | None]
    weight_kg: Mapped[float | None]
    injury_notes: Mapped[str | None]
    field_sources: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(_UTCDateTime)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str]
    title: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(_UTCDateTime)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(primary_key=True)
    conversation_id: Mapped[str]
    role: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime)
