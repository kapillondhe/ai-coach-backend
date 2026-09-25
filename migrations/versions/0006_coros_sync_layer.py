"""coros sync layer: integration_sync_state, synced_activities, synced_daily_metrics, synced_snapshots

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-24

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "integration_sync_state",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("provider", sa.Text(), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="idle"),
        sa.Column("last_synced_at", sa.DateTime(), nullable=True),
        sa.Column("last_backfill_completed_at", sa.DateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
    )

    op.create_table(
        "synced_activities",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("discipline", sa.Text(), nullable=False),
        sa.Column("sport_type_code", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("distance_km", sa.Float(), nullable=True),
        sa.Column("avg_pace_sec_per_km", sa.Integer(), nullable=True),
        sa.Column("avg_hr", sa.Integer(), nullable=True),
        sa.Column("calories", sa.Integer(), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("synced_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("user_id", "provider", "external_id", name="synced_activities_external_uq"),
    )
    op.create_index("synced_activities_user_id_idx", "synced_activities", ["user_id", "provider"])
    op.create_index("synced_activities_started_at_idx", "synced_activities", ["started_at"])

    op.create_table(
        "synced_daily_metrics",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("metric_type", sa.Text(), nullable=False),
        sa.Column("date", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("synced_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("user_id", "provider", "metric_type", "date", name="synced_daily_metrics_uq"),
    )
    op.create_index(
        "synced_daily_metrics_user_id_idx", "synced_daily_metrics", ["user_id", "provider", "metric_type"]
    )

    op.create_table(
        "synced_snapshots",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("snapshot_type", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("captured_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index(
        "synced_snapshots_user_id_idx", "synced_snapshots", ["user_id", "provider", "snapshot_type", "captured_at"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("synced_snapshots_user_id_idx", table_name="synced_snapshots")
    op.drop_table("synced_snapshots")
    op.drop_index("synced_daily_metrics_user_id_idx", table_name="synced_daily_metrics")
    op.drop_table("synced_daily_metrics")
    op.drop_index("synced_activities_started_at_idx", table_name="synced_activities")
    op.drop_index("synced_activities_user_id_idx", table_name="synced_activities")
    op.drop_table("synced_activities")
    op.drop_table("integration_sync_state")
