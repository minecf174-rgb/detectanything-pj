from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.models.models import Model


class ModelRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, model_id: int) -> Model | None:
        return self.db.scalar(
            select(Model)
            .where(Model.model_id == model_id, Model.deleted_at.is_(None))
            .options(selectinload(Model.artifacts))
        )

    def list_by_project(self, project_id: int) -> list[Model]:
        return list(
            self.db.scalars(
                select(Model)
                .where(Model.project_id == project_id, Model.deleted_at.is_(None))
                .options(selectinload(Model.artifacts))
                .order_by(Model.version)
            )
        )

    def next_version(self, project_id: int) -> int:
        current = self.db.scalar(select(func.max(Model.version)).where(Model.project_id == project_id))
        return (current or 0) + 1
