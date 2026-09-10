from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.session import Base
from app.core.database.types import PrimaryKeyInt

if TYPE_CHECKING:
    from app.modules.classes.models import LabelClass
    from app.modules.datasets.models import DatasetVersion
    from app.modules.images.models import Image
    from app.modules.models.models import Model
    from app.modules.training.models import TrainingRun


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    username: Mapped[str | None] = mapped_column(String)
    password_hash: Mapped[str | None] = mapped_column(String)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    projects: Mapped[list["Project"]] = relationship(back_populates="owner")


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (Index("ix_projects_owner_id", "owner_id"),)

    project_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    owner: Mapped[User] = relationship(back_populates="projects")
    classes: Mapped[list["LabelClass"]] = relationship(back_populates="project")
    images: Mapped[list["Image"]] = relationship(back_populates="project")
    dataset_versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="project")
    training_runs: Mapped[list["TrainingRun"]] = relationship(back_populates="project")
    models: Mapped[list["Model"]] = relationship(back_populates="project")
