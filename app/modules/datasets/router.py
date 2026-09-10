from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.datasets.schemas import (
    DatasetBuildRead,
    DatasetExportRead,
    DatasetVersionCreate,
    DatasetVersionRead,
)
from app.modules.datasets.service import DatasetService
from app.modules.projects.models import User

router = APIRouter()


@router.post("/versions", response_model=DatasetVersionRead)
def create_dataset_version(
    payload: DatasetVersionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return DatasetService(db).create_dataset_version(payload, current_user)


@router.get("/projects/{project_id}/versions", response_model=list[DatasetVersionRead])
def list_project_dataset_versions(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return DatasetService(db).list_project_versions(project_id, current_user)


@router.delete("/versions/{version_id}", status_code=204)
def delete_dataset_version(
    version_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    DatasetService(db).delete_dataset_version(version_id, current_user)


@router.post("/versions/{version_id}/build-yolo", response_model=DatasetBuildRead)
def build_yolo_dataset(
    version_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return DatasetService(db).build_yolo_dataset(version_id, current_user)


@router.post("/versions/{version_id}/export-yolo", response_model=DatasetExportRead)
def export_yolo_dataset(
    version_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return DatasetService(db).export_yolo_dataset(version_id, current_user)


@router.get("/versions/{version_id}/export-yolo/download")
def download_yolo_dataset_export(
    version_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    path = DatasetService(db).get_yolo_export_file_path(version_id, current_user)
    return FileResponse(path, filename=path.name)
