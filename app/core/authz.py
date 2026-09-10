from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.projects.models import Project, User


def require_project_owner(db: Session, user: User, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != user.user_id:
        raise HTTPException(status_code=403, detail="You do not have access to this project")
    return project
