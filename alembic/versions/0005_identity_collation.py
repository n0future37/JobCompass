"""来源ID按原始大小写精确区分，避免数据库默认排序规则误合并。"""

from alembic import op
import sqlalchemy as sa

revision = "0005_identity_collation"
down_revision = "0004_versions"
branch_labels = None
depends_on = None


def upgrade():
    for table, field, length in [
        ("rag_jobs", "job_key", 192),
        ("rag_jobs", "source_job_id", 128),
        ("rag_versions", "job_key", 192),
    ]:
        op.alter_column(
            table, field, type_=sa.String(length, collation="utf8mb4_bin"), existing_nullable=False
        )


def downgrade():
    raise RuntimeError("大小写不同的来源ID可能并存，禁止有损排序规则降级")
