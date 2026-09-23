"""profiles: drop target_race, target_distance, target_date columns

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-23

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("profiles", "target_race")
    op.drop_column("profiles", "target_distance")
    op.drop_column("profiles", "target_date")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column("profiles", sa.Column("target_race", sa.Text(), nullable=True))
    op.add_column("profiles", sa.Column("target_distance", sa.Text(), nullable=True))
    op.add_column("profiles", sa.Column("target_date", sa.Text(), nullable=True))
