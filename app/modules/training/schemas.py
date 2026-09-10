from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrainingStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


SUPPORTED_YOLO_DETECT_BASE_MODELS = {
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
}

SUPPORTED_YOLO_POSE_BASE_MODELS = {
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
}


class YoloTrainingConfig(BaseModel):
    epochs: int = Field(default=50, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=320, le=2048)
    batch: int = Field(default=4, ge=-1, le=128)
    patience: int = Field(default=25, ge=0, le=500)
    optimizer: Literal["auto", "SGD", "Adam", "AdamW"] = "auto"
    lr0: float = Field(default=0.01, gt=0, le=1)
    workers: int = Field(default=0, ge=0, le=32)
    device: Literal["auto", "mps", "cpu"] = "mps"
    cache: bool = False
    amp: bool = True
    cos_lr: bool = False
    val: bool = True
    plots: bool = True
    verbose: bool = True

    model_config = ConfigDict(extra="forbid")

    @field_validator("imgsz")
    @classmethod
    def validate_image_size(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError("imgsz must be divisible by 32")
        return value

    @field_validator("batch")
    @classmethod
    def validate_batch(cls, value: int) -> int:
        if value == 0:
            raise ValueError("batch must be -1 (auto) or greater than 0")
        return value


class TrainingRunCreate(BaseModel):
    project_id: int
    version_id: int
    model_type: Literal["YOLO26_DETECT", "YOLO26_POSE"] = "YOLO26_DETECT"
    base_model: str | None = "yolo26s.pt"
    training_config: YoloTrainingConfig | None = None

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, value: str | None) -> str:
        model = value or "yolo26s.pt"
        if model in SUPPORTED_YOLO_DETECT_BASE_MODELS | SUPPORTED_YOLO_POSE_BASE_MODELS:
            return model
        if model.startswith("model:") and model.removeprefix("model:").isdigit():
            return model
        raise ValueError("Unsupported base model")


class TrainingMetricRead(BaseModel):
    metric_id: int
    training_run_id: int
    epoch: int | None
    box_loss: float | None
    cls_loss: float | None
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingRunRead(BaseModel):
    training_run_id: int
    project_id: int
    version_id: int
    model_type: str
    base_model: str | None
    training_config: dict | None
    status: str
    progress_percent: float | None
    error_message: str | None
    log_object_key: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime | None
    metrics: list[TrainingMetricRead] = []

    model_config = {"from_attributes": True}


class ModelRead(BaseModel):
    model_id: int
    project_id: int
    training_run_id: int
    version: int
    weights_object_key: str
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingStartRead(BaseModel):
    training_run: TrainingRunRead
    model: ModelRead
    dataset_dir: str
