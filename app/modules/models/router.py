import json
from typing import Literal

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.models.schemas import (
    ModelBatchPredictionRead,
    ModelDetailRead,
    ModelPredictionRead,
    ModelRead,
)
from app.modules.models.service import ModelService
from app.modules.projects.models import User

router = APIRouter()


def _ndjson_response(events) -> StreamingResponse:
    return StreamingResponse(
        (json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/pose-pretrained/predict-batch", response_model=ModelBatchPredictionRead)
async def predict_with_pretrained_pose(
    project_id: int = Query(..., ge=1),
    files: list[UploadFile] = File(...),
    confidence_threshold: float = Query(default=0.25, ge=0.01, le=0.99),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    uploads = [(await file.read(), file.filename) for file in files]
    return ModelService(db).predict_pretrained_pose_images(
        project_id=project_id,
        uploads=uploads,
        confidence_threshold=confidence_threshold,
        current_user=current_user,
    )


@router.post("/pose-pretrained/predict-batch-stream")
async def stream_with_pretrained_pose(
    project_id: int = Query(..., ge=1),
    files: list[UploadFile] = File(...),
    confidence_threshold: float = Query(default=0.25, ge=0.01, le=0.99),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    uploads = [(await file.read(), file.filename) for file in files]
    events = ModelService(db).stream_pretrained_pose_images(
        project_id=project_id,
        uploads=uploads,
        confidence_threshold=confidence_threshold,
        current_user=current_user,
    )
    return _ndjson_response(events)


@router.post("/{model_id}/predict", response_model=ModelPredictionRead)
async def predict_with_model(
    model_id: int,
    file: UploadFile = File(...),
    confidence_threshold: float = Query(default=0.25, ge=0.01, le=0.99),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ModelService(db).predict_uploaded_image(
        model_id=model_id,
        image_data=await file.read(),
        filename=file.filename,
        confidence_threshold=confidence_threshold,
        current_user=current_user,
    )


@router.post("/{model_id}/predict-batch", response_model=ModelBatchPredictionRead)
async def predict_batch_with_model(
    model_id: int,
    files: list[UploadFile] = File(...),
    confidence_threshold: float = Query(default=0.25, ge=0.01, le=0.99),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    uploads = [(await file.read(), file.filename) for file in files]
    return ModelService(db).predict_uploaded_images(
        model_id=model_id,
        uploads=uploads,
        confidence_threshold=confidence_threshold,
        current_user=current_user,
    )


@router.post("/{model_id}/predict-batch-stream")
async def stream_batch_with_model(
    model_id: int,
    files: list[UploadFile] = File(...),
    confidence_threshold: float = Query(default=0.25, ge=0.01, le=0.99),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    uploads = [(await file.read(), file.filename) for file in files]
    events = ModelService(db).stream_uploaded_images(
        model_id=model_id,
        uploads=uploads,
        confidence_threshold=confidence_threshold,
        current_user=current_user,
    )
    return _ndjson_response(events)


@router.get("/project/{project_id}", response_model=list[ModelRead])
def list_project_models(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ModelService(db).list_project_models(project_id, current_user)


@router.get("/{model_id}", response_model=ModelDetailRead)
def get_model(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ModelService(db).get_model(model_id, current_user)


@router.get("/{model_id}/download")
def download_model_weights(
    model_id: int,
    format: Literal["pt", "onnx", "torchscript"] = Query(default="pt"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    path = ModelService(db).get_download_file_path(model_id, format, current_user)
    return FileResponse(path, filename=path.name)


@router.delete("/{model_id}", status_code=204)
def delete_model(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ModelService(db).delete_model(model_id, current_user)
    return Response(status_code=204)
