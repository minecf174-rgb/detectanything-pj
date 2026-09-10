from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.images.schemas import (
    ArchiveInspectionRead,
    ImageBulkDeleteRead,
    ImageBulkDeleteRequest,
    ImageBulkStatusRead,
    ImageBulkStatusRequest,
    ImageRead,
    YoloImportRead,
)
from app.modules.images.service import ImageService
from app.modules.projects.models import User

router = APIRouter()


@router.post("/upload", response_model=ImageRead)
async def upload_image(
    project_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await ImageService(db).upload_image(project_id, file, current_user)


@router.post("/import-yolo", response_model=YoloImportRead)
async def import_yolo_dataset(
    project_id: int,
    class_names: str | None = None,
    class_indexes: str | None = None,
    max_images: int | None = None,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await ImageService(db).import_yolo_zip(
        project_id, current_user, file, class_names, max_images, class_indexes
    )


@router.post("/inspect-archive", response_model=ArchiveInspectionRead)
async def inspect_archive(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await ImageService(db).inspect_archive(file)


@router.post("/import-image-archive", response_model=YoloImportRead)
async def import_image_archive(
    project_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await ImageService(db).import_image_archive(project_id, current_user, file)


@router.get("/project/{project_id}", response_model=list[ImageRead])
def list_project_images(
    project_id: int,
    include_archived: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ImageService(db).list_project_images(project_id, include_archived, current_user)


@router.post("/bulk-status", response_model=ImageBulkStatusRead)
def update_images_status(
    payload: ImageBulkStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ImageService(db).update_images_status(payload, current_user)


@router.post("/bulk-delete", response_model=ImageBulkDeleteRead)
def delete_images_permanently(
    payload: ImageBulkDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ImageService(db).delete_images_permanently(payload, current_user)


@router.get("/{image_id}/file")
def get_image_file(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    path = ImageService(db).get_image_file_path(image_id, current_user)
    return FileResponse(
        path,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
