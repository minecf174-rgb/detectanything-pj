from datetime import datetime
from enum import StrEnum

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.modules.annotations.schemas import AnnotationSetRead


class AutoLabelStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AutoLabelItemStatus(StrEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVIEWED = "REVIEWED"


class AutoLabelJobCreate(BaseModel):
    project_id: int
    model_id: int | None = None
    base_model: Literal["yolo26n-pose.pt"] | None = None
    image_ids: list[int] = Field(min_length=1)
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    max_predictions_per_image: int | None = Field(default=20, ge=1, le=300)
    only_unlabeled: bool = True

    @model_validator(mode="after")
    def validate_model_source(self):
        if (self.model_id is None) == (self.base_model is None):
            raise ValueError("Choose exactly one trained model or pretrained base model")
        return self


class AutoLabelJobItemRead(BaseModel):
    job_id: int
    image_id: int
    annotation_set_id: int | None
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class AutoLabelJobRead(BaseModel):
    job_id: int
    project_id: int
    model_id: int | None
    base_model: str | None
    confidence_threshold: float | None
    max_predictions_per_image: int | None
    total_images: int | None
    processed_images: int | None
    status: str
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime | None
    items: list[AutoLabelJobItemRead] = []

    model_config = {"from_attributes": True}


class AutoLabelStartRead(BaseModel):
    job: AutoLabelJobRead
    suggestions: list[AnnotationSetRead]
