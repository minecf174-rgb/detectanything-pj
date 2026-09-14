from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.modules.classes.repository import ClassRepository
from app.core.authz import require_project_owner
from app.modules.classes.schemas import ClassCreate, ClassUpdate
from app.modules.projects.models import User


class ClassService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = ClassRepository(db)

    def create_class(self, data: ClassCreate, current_user: User):
        require_project_owner(self.db, current_user, data.project_id)
        try:
            label_class = self.repository.create(data)
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            raise HTTPException(status_code=409, detail="Class name or index already exists") from exc
        self.db.refresh(label_class)
        return label_class

    def list_project_classes(self, project_id: int, current_user: User):
        require_project_owner(self.db, current_user, project_id)
        return self.repository.list_by_project(project_id)

    def update_class(self, class_id: int, data: ClassUpdate, current_user: User):
        label_class = self.repository.get(class_id)
        if label_class is None:
            raise HTTPException(status_code=404, detail="Class not found")
        require_project_owner(self.db, current_user, label_class.project_id)
        try:
            self.repository.update(label_class, data)
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            raise HTTPException(status_code=409, detail="Class name already exists") from exc
        self.db.refresh(label_class)
        return label_class

    def delete_class(self, class_id: int, current_user: User) -> None:
        label_class = self.repository.get(class_id)
        if label_class is None:
            raise HTTPException(status_code=404, detail="Class not found")
        project_id = label_class.project_id
        require_project_owner(self.db, current_user, project_id)
        self.repository.delete_annotations_for_class(class_id)
        self.repository.delete(label_class)
        self.db.flush()
        for index, remaining_class in enumerate(self.repository.list_by_project(project_id)):
            remaining_class.class_index = index
        self.db.commit()
