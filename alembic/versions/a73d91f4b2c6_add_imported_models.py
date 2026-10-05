"""Allow project-scoped imported models with class metadata.

Revision ID: a73d91f4b2c6
Revises: 4e7b93c0d12a
"""

import sqlalchemy as sa

from alembic import op

revision = "a73d91f4b2c6"
down_revision = "4e7b93c0d12a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("models") as batch:
        batch.add_column(sa.Column("uploaded_by_user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=True))
        batch.add_column(sa.Column("source_type", sa.String(), nullable=False, server_default="TRAINED"))
        batch.add_column(sa.Column("model_type", sa.String(), nullable=True))
        batch.add_column(sa.Column("class_names", sa.JSON(), nullable=True))
        batch.create_foreign_key("fk_models_uploaded_by_user_id", "users", ["uploaded_by_user_id"], ["user_id"])
    op.execute(
        "UPDATE models SET model_type = (SELECT training_runs.model_type FROM training_runs "
        "WHERE training_runs.training_run_id = models.training_run_id)"
    )
    with op.batch_alter_table("models") as batch:
        batch.alter_column("model_type", nullable=False)
        batch.alter_column("training_run_id", nullable=True)


def downgrade() -> None:
    # Imported models cannot be represented by the old non-null training_run_id schema.
    imported_count = op.get_bind().scalar(sa.text("SELECT COUNT(*) FROM models WHERE training_run_id IS NULL"))
    if imported_count:
        raise RuntimeError("Remove imported models before downgrading this migration")
    with op.batch_alter_table("models") as batch:
        batch.alter_column("training_run_id", nullable=False)
        batch.drop_constraint("fk_models_uploaded_by_user_id", type_="foreignkey")
        batch.drop_column("class_names")
        batch.drop_column("model_type")
        batch.drop_column("source_type")
        batch.drop_column("uploaded_by_user_id")
