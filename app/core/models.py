"""ORM models: Alembic's `target_metadata`.

These must describe the schema the migrations in `migrations/versions` actually
created (TEXT strings, naive `timestamp` columns, `json` not `jsonb`, the named
indexes, the conversation_messages FK with ON DELETE CASCADE), so `alembic check`
reports no drift and `alembic revision --autogenerate` only picks up real changes.
"""

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Text,
    TypeDecorator,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Server-side defaults as the migrations declared them. Every service write sets
# these columns explicitly; they're here so the models mirror the real schema.
_NOW = text("CURRENT_TIMESTAMP")
_EMPTY_JSON = "{}"


class SyncStatusValue(StrEnum):
    """`integration_sync_state.status` values. Stored as plain TEXT."""

    IDLE = "idle"
    SYNCING = "syncing"
    ERROR = "error"


class Discipline(StrEnum):
    """`synced_activities.discipline` values. Stored as plain TEXT."""

    RUN = "run"
    BIKE = "bike"
    SWIM = "swim"
    OTHER = "other"


class _UTCDateTime(TypeDecorator):
    """Aware-UTC datetimes in Python, naive UTC `timestamp without time zone` in Postgres."""

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
    # The migrations create every string column as TEXT, not VARCHAR.
    type_annotation_map = {str: Text}


class OAuthClient(Base):
    __tablename__ = "oauth_clients"

    provider: Mapped[str] = mapped_column(primary_key=True)
    client_id: Mapped[str]
    client_secret: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)


class OAuthState(Base):
    __tablename__ = "oauth_states"
    __table_args__ = (Index("oauth_states_expires_at_idx", "expires_at"),)

    state: Mapped[str] = mapped_column(primary_key=True)
    provider: Mapped[str]
    user_id: Mapped[str]
    code_verifier: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)
    expires_at: Mapped[datetime] = mapped_column(_UTCDateTime)


class UserIntegration(Base):
    __tablename__ = "user_integrations"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(primary_key=True)
    access_token_encrypted: Mapped[str]
    refresh_token_encrypted: Mapped[str | None]
    scope: Mapped[str | None]
    expires_at: Mapped[datetime | None] = mapped_column(_UTCDateTime)
    connected_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)
    updated_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)


class Profile(Base):
    __tablename__ = "profiles"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str | None]
    weight_kg: Mapped[float | None]
    injury_notes: Mapped[str | None]
    field_sources: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_EMPTY_JSON)
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)
    updated_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("conversations_user_id_idx", "user_id"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str]
    title: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)
    updated_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (Index("conversation_messages_conversation_id_idx", "conversation_id"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    # Deleting a conversation deletes its messages in the database.
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)


class IntegrationSyncState(Base):
    __tablename__ = "integration_sync_state"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(primary_key=True)
    # Explicit Text: an enum annotation would otherwise map to a Postgres ENUM type.
    status: Mapped[SyncStatusValue] = mapped_column(Text, server_default=SyncStatusValue.IDLE.value)
    last_synced_at: Mapped[datetime | None] = mapped_column(_UTCDateTime)
    last_backfill_completed_at: Mapped[datetime | None] = mapped_column(_UTCDateTime)
    last_error: Mapped[str | None]


class SyncedActivity(Base):
    __tablename__ = "synced_activities"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", "external_id", name="synced_activities_external_uq"),
        Index("synced_activities_user_id_idx", "user_id", "provider"),
        Index("synced_activities_started_at_idx", "started_at"),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str]
    provider: Mapped[str]
    external_id: Mapped[str]  # COROS labelId
    discipline: Mapped[Discipline] = mapped_column(Text)
    sport_type_code: Mapped[int]
    started_at: Mapped[datetime | None] = mapped_column(_UTCDateTime)
    ended_at: Mapped[datetime | None] = mapped_column(_UTCDateTime)
    duration_seconds: Mapped[int | None]
    distance_km: Mapped[float | None]
    avg_pace_sec_per_km: Mapped[int | None]
    avg_hr: Mapped[int | None]
    calories: Mapped[int | None]
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_EMPTY_JSON)
    synced_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)


class SyncedDailyMetric(Base):
    __tablename__ = "synced_daily_metrics"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", "metric_type", "date", name="synced_daily_metrics_uq"),
        Index("synced_daily_metrics_user_id_idx", "user_id", "provider", "metric_type"),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str]
    provider: Mapped[str]
    metric_type: Mapped[str]  # resting_hr | avg_hr | training_load
    date: Mapped[str]  # yyyy-mm-dd
    value: Mapped[float | None]
    extra: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_EMPTY_JSON)
    synced_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)


class UserMemory(Base):
    __tablename__ = "user_memories"
    __table_args__ = (Index("user_memories_user_id_idx", "user_id"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)


class SyncedSnapshot(Base):
    __tablename__ = "synced_snapshots"
    __table_args__ = (Index("synced_snapshots_user_id_idx", "user_id", "provider", "snapshot_type", "captured_at"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str]
    provider: Mapped[str]
    snapshot_type: Mapped[str]  # recovery_status | fitness_assessment
    data: Mapped[dict] = mapped_column(JSON, default=dict, server_default=_EMPTY_JSON)
    captured_at: Mapped[datetime] = mapped_column(_UTCDateTime, server_default=_NOW)
