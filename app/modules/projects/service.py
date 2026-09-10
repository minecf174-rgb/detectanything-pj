from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config.settings import get_settings
from app.core.authz import require_project_owner
from app.core.storage.service import LocalStorageService
from app.modules.projects.models import User
from app.modules.projects.repository import ProjectRepository
from app.modules.projects.schemas import ProjectCreate


class ProjectService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = ProjectRepository(db)
        self.storage = LocalStorageService(get_settings().storage_root)

    def create_project(self, data: ProjectCreate, current_user: User):
        project = self.repository.create(data, current_user.user_id)
        self.db.commit()
        self.db.refresh(project)
        return project

    def list_projects(self, current_user: User, owner_id: int | None = None):
        if owner_id is not None and owner_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="You can only list your own projects")
        return self.repository.list_by_owner(current_user.user_id)

    def get_project(self, project_id: int, current_user: User):
        return require_project_owner(self.db, current_user, project_id)

    def delete_project(self, project_id: int, current_user: User) -> None:
        require_project_owner(self.db, current_user, project_id)
        self.repository.delete_project_tree(project_id)
        self.db.commit()
        self.storage.delete_prefix(f"projects/project_{project_id}")
