"""Add explicit verified-negative state to annotation sets.

Revision ID: 4e7b93c0d12a
Revises: 3d8f16a4c2e9
"""

import sqlalchemy as sa
from alembic import op

revision = "4e7b93c0d12a"
down_revision = "3d8f16a4c2e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("annotation_sets", sa.Column("is_negative", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("annotation_sets", "is_negative")
