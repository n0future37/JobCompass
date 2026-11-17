"""账号私有的岗位偏好和投递记录。"""

from alembic import op
import sqlalchemy as sa

revision = "0006_career"
down_revision = "0005_identity_collation"
branch_labels = depends_on = None


def upgrade():
    for name, fields in [
        (
            "rag_job_preferences",
            [
                sa.Column("preference", sa.String(16), nullable=False),
                sa.Column("note", sa.String(1000), nullable=False),
            ],
        ),
        (
            "rag_applications",
            [
                sa.Column("status", sa.String(20), nullable=False),
                sa.Column("note", sa.String(1000), nullable=False),
                sa.Column("version", sa.Integer(), nullable=False),
                sa.Column("snapshot", sa.JSON(), nullable=False),
            ],
        ),
    ]:
        op.create_table(
            name,
            sa.Column(
                "owner", sa.String(36), sa.ForeignKey("rag_users.id", ondelete="CASCADE"), primary_key=True
            ),
            sa.Column(
                "job_key",
                sa.String(192, collation="utf8mb4_bin"),
                sa.ForeignKey("rag_jobs.job_key"),
                primary_key=True,
            ),
            *fields,
            sa.Column("updated_at", sa.String(40), nullable=False),
        )


def downgrade():
    op.drop_table("rag_applications")
    op.drop_table("rag_job_preferences")
