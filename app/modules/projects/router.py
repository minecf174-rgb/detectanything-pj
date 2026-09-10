from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.projects.models import User
from app.modules.projects.schemas import ProjectCreate, ProjectRead
from app.modules.projects.service import ProjectService

router = APIRouter()


@router.post("", response_model=ProjectRead)
def create_project(
    payload: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ProjectService(db).create_project(payload, current_user)


@router.get("", response_model=list[ProjectRead])
def list_projects(
    owner_id: int | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ProjectService(db).list_projects(current_user, owner_id)


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ProjectService(db).get_project(project_id, current_user)


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ProjectService(db).delete_project(project_id, current_user)
