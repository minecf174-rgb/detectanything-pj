from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.classes.schemas import ClassCreate, ClassRead, ClassUpdate
from app.modules.classes.service import ClassService
from app.modules.projects.models import User

router = APIRouter()


@router.post("", response_model=ClassRead)
def create_class(
    payload: ClassCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ClassService(db).create_class(payload, current_user)


@router.get("/project/{project_id}", response_model=list[ClassRead])
def list_project_classes(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ClassService(db).list_project_classes(project_id, current_user)


@router.patch("/{class_id}", response_model=ClassRead)
def update_class(
    class_id: int,
    payload: ClassUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ClassService(db).update_class(class_id, payload, current_user)


@router.delete("/{class_id}", status_code=204)
def delete_class(
    class_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ClassService(db).delete_class(class_id, current_user)
