from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.annotations.schemas import (
    AnnotationBulkAcceptRead,
    AnnotationBulkReviewRead,
    AnnotationBulkReviewRequest,
    AnnotationCorrectionRequest,
    AnnotationReviewRequest,
    AnnotationSetCreate,
    AnnotationSetRead,
    ImageAnnotationStatusRead,
)
from app.modules.annotations.service import AnnotationService
from app.modules.projects.models import User

router = APIRouter()


@router.get("/projects/{project_id}/image-statuses", response_model=list[ImageAnnotationStatusRead])
def list_project_image_annotation_statuses(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).list_project_image_statuses(project_id, current_user)


@router.post("/images/{image_id}/annotation-sets", response_model=AnnotationSetRead)
def create_annotation_set(
    image_id: int,
    payload: AnnotationSetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).create_annotation_set(image_id, payload, current_user)


@router.get("/images/{image_id}/annotation-sets", response_model=list[AnnotationSetRead])
def list_image_annotation_sets(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).list_image_annotation_sets(image_id, current_user)


@router.post("/sets/{annotation_set_id}/accept", response_model=AnnotationSetRead)
def accept_annotation_set(
    annotation_set_id: int,
    payload: AnnotationReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).accept_annotation_set(annotation_set_id, payload, current_user)


@router.post("/sets/{annotation_set_id}/reject", response_model=AnnotationSetRead)
def reject_annotation_set(
    annotation_set_id: int,
    payload: AnnotationReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).reject_annotation_set(annotation_set_id, payload, current_user)


@router.post("/sets/reject-bulk", response_model=AnnotationBulkReviewRead)
def reject_annotation_sets_bulk(
    payload: AnnotationBulkReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).reject_annotation_sets_bulk(payload, current_user)


@router.post("/sets/accept-bulk", response_model=AnnotationBulkAcceptRead)
def accept_annotation_sets_bulk(
    payload: AnnotationBulkReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).accept_annotation_sets_bulk(payload, current_user)


@router.post("/sets/{annotation_set_id}/correct", response_model=AnnotationSetRead)
def correct_annotation_set(
    annotation_set_id: int,
    payload: AnnotationCorrectionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).correct_annotation_set(annotation_set_id, payload, current_user)


@router.post("/sets/{annotation_set_id}/edit", response_model=AnnotationSetRead)
def edit_approved_annotation_set(
    annotation_set_id: int,
    payload: AnnotationCorrectionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return AnnotationService(db).edit_approved_annotation_set(annotation_set_id, payload, current_user)
