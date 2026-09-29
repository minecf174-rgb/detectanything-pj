from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.projects.models import User
from app.modules.training.schemas import TrainingRunCreate, TrainingRunRead, TrainingStartRead
from app.modules.training.service import TrainingService

router = APIRouter()


@router.post("", response_model=TrainingRunRead)
def create_training_run(
    payload: TrainingRunCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return TrainingService(db).create_training_run(payload, current_user)


@router.get("/project/{project_id}", response_model=list[TrainingRunRead])
def list_project_training_runs(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return TrainingService(db).list_project_runs(project_id, current_user)


@router.post("/{training_run_id}/start-local-mock", response_model=TrainingStartRead)
def start_local_mock_training(
    training_run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return TrainingService(db).start_local_mock(training_run_id, current_user)


@router.post("/{training_run_id}/start-local-yolo", response_model=TrainingStartRead)
def start_local_yolo_training(
    training_run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return TrainingService(db).start_local_yolo(training_run_id, current_user)


@router.post("/{training_run_id}/resume-local-yolo", response_model=TrainingStartRead)
def resume_local_yolo_training(
    training_run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return TrainingService(db).resume_local_yolo(training_run_id, current_user)
