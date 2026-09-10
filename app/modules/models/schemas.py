from datetime import datetime

from pydantic import BaseModel


class ModelArtifactRead(BaseModel):
    artifact_id: int
    model_id: int
    artifact_type: str
    object_key: str
    created_at: datetime

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


class ModelDetailRead(ModelRead):
    artifacts: list[ModelArtifactRead]
    weights_exists: bool


class ModelPredictionKeypointRead(BaseModel):
    x: float
    y: float
    visibility: int


class ModelPredictionBoxRead(BaseModel):
    class_id: int
    class_index: int
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    keypoints: list[ModelPredictionKeypointRead] = []


class ModelPredictionRead(BaseModel):
    model_id: int
    project_id: int
    image_width: int
    image_height: int
    boxes: list[ModelPredictionBoxRead]


class ModelBatchPredictionRead(BaseModel):
    model_id: int
    project_id: int
    results: list[ModelPredictionRead]
