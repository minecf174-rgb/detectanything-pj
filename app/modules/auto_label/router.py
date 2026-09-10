from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.auto_label.schemas import AutoLabelJobCreate, AutoLabelJobRead, AutoLabelStartRead
from app.modules.auto_label.service import AutoLabelService
from app.modules.projects.models import User

router = APIRouter()


@router.post("", response_model=AutoLabelJobRead)
def create_auto_label_job(
    payload: AutoLabelJobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AutoLabelService(db).create_job(payload, current_user)


@router.get("/project/{project_id}", response_model=list[AutoLabelJobRead])
def list_project_auto_label_jobs(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AutoLabelService(db).list_project_jobs(project_id, current_user)


@router.get("/{job_id}", response_model=AutoLabelJobRead)
def get_auto_label_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AutoLabelService(db).get_job(job_id, current_user)


@router.post("/{job_id}/start-local-mock", response_model=AutoLabelStartRead)
def start_local_mock_auto_label(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AutoLabelService(db).start_local_mock(job_id, current_user)


@router.post("/{job_id}/start-local-yolo", response_model=AutoLabelStartRead)
def start_local_yolo_auto_label(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AutoLabelService(db).start_local_yolo(job_id, current_user)


@router.post("/{job_id}/start-local-yolo-async", response_model=AutoLabelJobRead)
def start_local_yolo_auto_label_async(
    job_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AutoLabelService(db).start_local_yolo_async(job_id, background_tasks, current_user)
