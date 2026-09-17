"""add model soft delete

Revision ID: 3d8f16a4c2e9
Revises: 91ce4a28d6bf
Create Date: 2026-09-14
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "3d8f16a4c2e9"
down_revision: Union[str, Sequence[str], None] = "91ce4a28d6bf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("models", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "deleted_at")
