from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    username: str | None = None


class UserRead(BaseModel):
    user_id: int
    email: EmailStr
    username: str | None
    created_at: datetime

    model_config = {"from_attributes": True}

