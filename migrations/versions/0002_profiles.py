"""profiles: signed-in user's editable facts

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "profiles",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("weight_kg", sa.Float(), nullable=True),
        sa.Column("target_race", sa.Text(), nullable=True),
        sa.Column("target_distance", sa.Text(), nullable=True),
        sa.Column("target_date", sa.Text(), nullable=True),
        sa.Column("injury_notes", sa.Text(), nullable=True),
        sa.Column("theme", sa.Text(), nullable=False, server_default="system"),
        sa.Column("field_sources", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("profiles")
