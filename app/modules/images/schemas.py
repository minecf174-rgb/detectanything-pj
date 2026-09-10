from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ImageRead(BaseModel):
    image_id: int
    project_id: int
    original_filename: str | None
    object_key: str
    file_hash: str | None
    width: int | None
    height: int | None
    status: str
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class YoloImportRead(BaseModel):
    imported_images: int
    imported_annotation_sets: int
    imported_boxes: int
    classes: list[str]


class ArchiveInspectionRead(BaseModel):
    archive_type: Literal["YOLO", "IMAGES"]
    image_count: int
    class_names: list[str]


class ImageBulkStatusRequest(BaseModel):
    project_id: int
    image_ids: list[int] = Field(min_length=1, max_length=10000)
    status: Literal["UPLOADED", "ARCHIVED"]


class ImageBulkStatusRead(BaseModel):
    updated_count: int
    status: str


class ImageBulkDeleteRequest(BaseModel):
    project_id: int
    image_ids: list[int] = Field(min_length=1, max_length=10000)


class ImageBulkDeleteRead(BaseModel):
    deleted_count: int
