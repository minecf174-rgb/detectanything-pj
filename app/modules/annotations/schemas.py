from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class AnnotationSourceType(StrEnum):
    HUMAN = "HUMAN"
    MODEL = "MODEL"
    HUMAN_CORRECTED = "HUMAN_CORRECTED"


class AnnotationStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class AnnotationCreate(BaseModel):
    class_id: int
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    keypoints: list["AnnotationKeypointCreate"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_box(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Bounding box must satisfy x_min < x_max and y_min < y_max")
        return self


class AnnotationKeypointCreate(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    visibility: int = Field(default=2, ge=0, le=2)


class AnnotationSetCreate(BaseModel):
    annotations: list[AnnotationCreate]


class AnnotationReviewRequest(BaseModel):
    pass


class AnnotationBulkReviewRequest(BaseModel):
    annotation_set_ids: list[int] = Field(min_length=1, max_length=10000)


class AnnotationBulkReviewRead(BaseModel):
    rejected_count: int
    skipped_count: int


class AnnotationBulkAcceptRead(BaseModel):
    accepted_count: int
    skipped_count: int


class AnnotationCorrectionRequest(BaseModel):
    annotations: list[AnnotationCreate]


class AnnotationRead(BaseModel):
    annotation_id: int
    annotation_set_id: int
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float | None
    created_at: datetime
    keypoints: list["AnnotationKeypointRead"] = []

    model_config = {"from_attributes": True}


class AnnotationKeypointRead(BaseModel):
    keypoint_index: int
    x: float
    y: float
    visibility: int

    model_config = {"from_attributes": True}


class AnnotationSetRead(BaseModel):
    annotation_set_id: int
    image_id: int
    revision: int
    source_type: str
    source_model_id: int | None
    parent_annotation_set_id: int | None
    status: str
    created_by: int | None
    reviewed_by: int | None
    created_at: datetime
    reviewed_at: datetime | None
    annotations: list[AnnotationRead]

    model_config = {"from_attributes": True}


class ImageAnnotationStatusRead(BaseModel):
    image_id: int
    annotation_set_id: int
    source_type: AnnotationSourceType
    status: AnnotationStatus
    box_count: int
