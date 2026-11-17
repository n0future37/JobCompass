"""账号私有的简历原件与本地解析草稿。"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import MEDIUMBLOB

revision = "0007_resumes"
down_revision = "0006_career"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "rag_resumes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner", sa.String(36), sa.ForeignKey("rag_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("parser_version", sa.String(40), nullable=False),
        sa.Column("filename", sa.String(200), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("original", MEDIUMBLOB(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.UniqueConstraint("owner", "sha256", "parser_version", name="uq_resume_owner_content_parser"),
    )
    op.create_index("ix_rag_resumes_owner", "rag_resumes", ["owner"])


def downgrade():
    op.drop_table("rag_resumes")
