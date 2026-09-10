"""add pretrained auto label source

Revision ID: 91ce4a28d6bf
Revises: 7a1b3d9e5f24
Create Date: 2026-09-01
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "91ce4a28d6bf"
down_revision: Union[str, Sequence[str], None] = "7a1b3d9e5f24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("auto_label_jobs") as batch_op:
        batch_op.alter_column("model_id", existing_type=sa.BigInteger(), nullable=True)
        batch_op.add_column(sa.Column("base_model", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("auto_label_jobs") as batch_op:
        batch_op.drop_column("base_model")
        batch_op.alter_column("model_id", existing_type=sa.BigInteger(), nullable=False)
