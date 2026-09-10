from datetime import datetime

from pydantic import BaseModel


class ClassCreate(BaseModel):
    project_id: int
    name: str
    class_index: int
    color: str = "#16a34a"


class ClassRead(BaseModel):
    class_id: int
    project_id: int
    name: str
    class_index: int
    color: str
    created_at: datetime

    model_config = {"from_attributes": True}
