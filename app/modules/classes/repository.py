from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.modules.annotations.models import Annotation
from app.modules.classes.models import LabelClass
from app.modules.classes.schemas import ClassCreate, ClassUpdate


class ClassRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, data: ClassCreate) -> LabelClass:
        label_class = LabelClass(
            project_id=data.project_id,
            name=data.name,
            class_index=data.class_index,
            color=data.color,
        )
        self.db.add(label_class)
        self.db.flush()
        return label_class

    def list_by_project(self, project_id: int) -> list[LabelClass]:
        return list(
            self.db.scalars(
                select(LabelClass)
                .where(LabelClass.project_id == project_id)
                .order_by(LabelClass.class_index)
            )
        )

    def get(self, class_id: int) -> LabelClass | None:
        return self.db.get(LabelClass, class_id)

    def update(self, label_class: LabelClass, data: ClassUpdate) -> LabelClass:
        label_class.name = data.name
        label_class.color = data.color.lower()
        self.db.flush()
        return label_class

    def annotation_count(self, class_id: int) -> int:
        return self.db.scalar(
            select(func.count()).select_from(Annotation).where(Annotation.class_id == class_id)
        ) or 0

    def delete_annotations_for_class(self, class_id: int) -> None:
        self.db.execute(delete(Annotation).where(Annotation.class_id == class_id))

    def delete(self, label_class: LabelClass) -> None:
        self.db.delete(label_class)
