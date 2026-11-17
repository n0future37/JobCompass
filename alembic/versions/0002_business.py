"""账号、画像、会话、任务租约和岗位归属。"""

from alembic import op
import sqlalchemy as sa

revision = "0002_business"
down_revision = "0001_rag"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "rag_jobs",
        sa.Column("owner", sa.String(64), nullable=False, server_default="system"),
    )
    op.create_table(
        "rag_users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("password_hash", sa.String(256), nullable=False),
        sa.Column("profile", sa.JSON(), nullable=False),
        sa.UniqueConstraint("username", name="uq_rag_username"),
    )
    op.create_table(
        "rag_sessions",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("rag_users.id"), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_rag_session_expiry", "rag_sessions", ["expires_at"])
    op.create_table(
        "rag_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner", sa.String(64), nullable=False),
        sa.Column("idempotency", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("available_at", sa.Float(), nullable=False),
        sa.Column("lease_until", sa.Float(), nullable=False),
        sa.Column("lease_token", sa.String(36)),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.UniqueConstraint("idempotency", name="uq_rag_task_key"),
    )
    op.create_index("ix_rag_task_claim", "rag_tasks", ["status", "available_at", "lease_until"])


def downgrade():
    for name in ("rag_tasks", "rag_sessions", "rag_users"):
        op.drop_table(name)
    op.drop_column("rag_jobs", "owner")
