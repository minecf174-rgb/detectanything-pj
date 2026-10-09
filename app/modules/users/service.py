from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.modules.projects.repository import ProjectRepository
from app.modules.projects.service import ProjectService
from app.modules.users.repository import UserRepository
from app.modules.users.schemas import UserCreate


class UserService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = UserRepository(db)
        self.project_repository = ProjectRepository(db)

    def create_user(self, data: UserCreate):
        try:
            user = self.repository.create(data)
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            raise HTTPException(status_code=409, detail="Email already exists") from exc
        self.db.refresh(user)
        return user

    def list_users(self):
        return self.repository.list()

    def delete_user(self, user_id: int) -> None:
        user = self.repository.get(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        project_service = ProjectService(self.db)
        project_ids = self.repository.project_ids_for_user(user_id)
        for project_id in project_ids:
            project_service.ensure_no_running_jobs(project_id)
        artifacts = {project_id: project_service.artifact_ids(project_id) for project_id in project_ids}
        for project_id in project_ids:
            self.project_repository.delete_project_tree(project_id)
        self.repository.clear_user_references(user_id)
        self.repository.delete(user)
        self.db.commit()
        for project_id, (version_ids, run_ids) in artifacts.items():
            project_service.cleanup_files(project_id, version_ids, run_ids)
