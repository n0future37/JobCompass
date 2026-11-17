"""按版本号保存每次变化；恢复为旧内容也保留新的版本事件。"""

from alembic import op
import sqlalchemy as sa

revision = "0004_versions"
down_revision = "0003_time_precision"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "rag_versions",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.execute(
        "UPDATE rag_versions SET version=CAST(JSON_UNQUOTE(JSON_EXTRACT(payload,'$.version')) AS UNSIGNED)"
    )
    op.drop_constraint("PRIMARY", "rag_versions", type_="primary")
    op.create_primary_key("PRIMARY", "rag_versions", ["job_key", "version"])
    op.create_index("ix_rag_versions_hash", "rag_versions", ["content_hash"])


def downgrade():
    raise RuntimeError("版本事件可能含重复内容摘要，禁止有损自动降级")
