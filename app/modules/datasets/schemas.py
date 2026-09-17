from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DatasetSplit(StrEnum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


class DatasetVersionCreate(BaseModel):
    project_id: int
    image_ids: list[int] | None = Field(default=None, min_length=1)
    preprocess_config: dict | None = None
    augmentation_config: dict | None = None
    default_split: DatasetSplit = DatasetSplit.TRAIN
    auto_split: bool = True


class DatasetVersionItemRead(BaseModel):
    version_id: int
    image_id: int
    annotation_set_id: int
    split: str

    model_config = {"from_attributes": True}


class DatasetVersionRead(BaseModel):
    version_id: int
    project_id: int
    version_number: int
    preprocess_config: dict | None
    augmentation_config: dict | None
    created_by: int | None
    created_at: datetime
    items: list[DatasetVersionItemRead]

    model_config = {"from_attributes": True}


class DatasetBuildRead(BaseModel):
    version_id: int
    output_dir: str
    data_yaml: str
    image_count: int
    label_count: int


class DatasetExportRead(BaseModel):
    version_id: int
    archive_path: str
