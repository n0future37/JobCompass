"""MySQL RAG业务表；首次迁移。"""

from alembic import op
import sqlalchemy as sa

revision = "0001_rag"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "rag_jobs",
        sa.Column("job_key", sa.String(192), primary_key=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("source_job_id", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.UniqueConstraint("source", "source_job_id", name="uq_rag_source_id"),
    )
    op.create_table(
        "rag_versions",
        sa.Column("job_key", sa.String(192), primary_key=True),
        sa.Column("content_hash", sa.String(64), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    for table, key in [("rag_imports", "fingerprint"), ("rag_indexes", "snapshot")]:
        op.create_table(
            table,
            sa.Column(key, sa.String(64), primary_key=True),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
        )
    op.create_table(
        "rag_history",
        sa.Column("query_id", sa.String(36), primary_key=True),
        sa.Column("owner", sa.String(64), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_rag_history_owner", "rag_history", ["owner"])
    op.create_table(
        "rag_feedback",
        sa.Column("feedback_id", sa.String(36), primary_key=True),
        sa.Column("query_id", sa.String(36), sa.ForeignKey("rag_history.query_id"), nullable=False),
        sa.Column("owner", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )


def downgrade():
    for name in ["rag_feedback", "rag_history", "rag_indexes", "rag_imports", "rag_versions", "rag_jobs"]:
        op.drop_table(name)
