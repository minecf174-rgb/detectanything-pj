from datetime import datetime

from typing import Literal

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    task_type: Literal["OBJECT_DETECTION", "POSE_ESTIMATION"] = "OBJECT_DETECTION"


class ProjectRead(BaseModel):
    project_id: int
    owner_id: int
    name: str
    task_type: str
    description: str | None
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}
