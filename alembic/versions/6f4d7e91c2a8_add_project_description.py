"""add project description

Revision ID: 6f4d7e91c2a8
Revises: 0f82d9b7a4b1
Create Date: 2026-08-31
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "6f4d7e91c2a8"
down_revision: Union[str, Sequence[str], None] = "0f82d9b7a4b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("description", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "description")
