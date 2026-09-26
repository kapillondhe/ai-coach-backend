"""user_memories: signed-in freeform, LLM-extracted memory store

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-25

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "user_memories",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("user_memories_user_id_idx", "user_memories", ["user_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("user_memories_user_id_idx", table_name="user_memories")
    op.drop_table("user_memories")
