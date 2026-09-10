"""add pose annotation keypoints

Revision ID: 7a1b3d9e5f24
Revises: 6f4d7e91c2a8
Create Date: 2026-08-31
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7a1b3d9e5f24"
down_revision: Union[str, Sequence[str], None] = "6f4d7e91c2a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "annotation_keypoints",
        sa.Column("keypoint_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True),
        sa.Column("annotation_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("keypoint_index", sa.Integer(), nullable=False),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("visibility", sa.Integer(), nullable=False, server_default="2"),
        sa.ForeignKeyConstraint(["annotation_id"], ["annotations.annotation_id"]),
        sa.UniqueConstraint("annotation_id", "keypoint_index", name="uq_annotation_keypoints_index"),
    )
    op.create_index("ix_annotation_keypoints_annotation_id", "annotation_keypoints", ["annotation_id"])


def downgrade() -> None:
    op.drop_index("ix_annotation_keypoints_annotation_id", table_name="annotation_keypoints")
    op.drop_table("annotation_keypoints")
