"""租约与过期时间采用DOUBLE，避免epoch秒在FLOAT中损失秒级精度。"""

from alembic import op
from sqlalchemy import Float

revision = "0003_time_precision"
down_revision = "0002_business"
branch_labels = None
depends_on = None


def upgrade():
    for table, field in [
        ("rag_sessions", "expires_at"),
        ("rag_tasks", "available_at"),
        ("rag_tasks", "lease_until"),
    ]:
        op.alter_column(
            table,
            field,
            type_=Float(53),
            existing_type=Float(),
            existing_nullable=False,
        )


def downgrade():
    for table, field in [
        ("rag_sessions", "expires_at"),
        ("rag_tasks", "available_at"),
        ("rag_tasks", "lease_until"),
    ]:
        op.alter_column(
            table,
            field,
            type_=Float(),
            existing_type=Float(53),
            existing_nullable=False,
        )
