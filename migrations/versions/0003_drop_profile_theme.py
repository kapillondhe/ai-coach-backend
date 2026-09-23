"""profiles: drop theme column (moved to frontend-only storage)

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-23

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("profiles", "theme")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "profiles",
        sa.Column("theme", sa.Text(), nullable=False, server_default="system"),
    )
