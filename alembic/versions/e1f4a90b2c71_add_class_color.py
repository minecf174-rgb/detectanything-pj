"""add_class_color

Revision ID: e1f4a90b2c71
Revises: bd5fd3455af4
Create Date: 2026-08-11 16:45:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "e1f4a90b2c71"
down_revision: Union[str, Sequence[str], None] = "bd5fd3455af4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "classes",
        sa.Column("color", sa.String(length=16), nullable=False, server_default="#16a34a"),
    )


def downgrade() -> None:
    op.drop_column("classes", "color")
