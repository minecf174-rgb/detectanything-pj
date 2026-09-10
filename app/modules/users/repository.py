from __future__ import annotations

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.modules.annotations.models import AnnotationSet
from app.modules.auth.models import AuthSession
from app.modules.datasets.models import DatasetVersion
from app.modules.projects.models import Project, User
from app.modules.users.schemas import UserCreate


class UserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, data: UserCreate) -> User:
        user = User(email=data.email, username=data.username)
        self.db.add(user)
        self.db.flush()
        return user

    def get(self, user_id: int) -> User | None:
        return self.db.get(User, user_id)

    def list(self) -> list[User]:
        return list(self.db.scalars(select(User).order_by(User.user_id)))

    def project_ids_for_user(self, user_id: int) -> list[int]:
        return list(self.db.scalars(select(Project.project_id).where(Project.owner_id == user_id)))

    def clear_user_references(self, user_id: int) -> None:
        self.db.execute(delete(AuthSession).where(AuthSession.user_id == user_id))
        self.db.execute(
            update(AnnotationSet)
            .where(AnnotationSet.created_by == user_id)
            .values(created_by=None)
        )
        self.db.execute(
            update(AnnotationSet)
            .where(AnnotationSet.reviewed_by == user_id)
            .values(reviewed_by=None)
        )
        self.db.execute(
            update(DatasetVersion)
            .where(DatasetVersion.created_by == user_id)
            .values(created_by=None)
        )

    def delete(self, user: User) -> None:
        self.db.delete(user)
