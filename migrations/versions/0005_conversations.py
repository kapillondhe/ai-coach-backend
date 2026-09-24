"""conversations: server-side persisted signed-in chat threads

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-23

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "conversations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("conversations_user_id_idx", "conversations", ["user_id"])

    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index(
        "conversation_messages_conversation_id_idx", "conversation_messages", ["conversation_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("conversation_messages_conversation_id_idx", table_name="conversation_messages")
    op.drop_table("conversation_messages")
    op.drop_index("conversations_user_id_idx", table_name="conversations")
    op.drop_table("conversations")
