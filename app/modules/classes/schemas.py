from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ClassCreate(BaseModel):
    project_id: int
    name: str
    class_index: int
    color: str = "#16a34a"


class ClassUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Class name cannot be blank")
        return normalized


class ClassRead(BaseModel):
    class_id: int
    project_id: int
    name: str
    class_index: int
    color: str
    created_at: datetime

    model_config = {"from_attributes": True}
