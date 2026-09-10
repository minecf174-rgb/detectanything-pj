from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.session import Base
from app.core.database.types import PrimaryKeyInt

if TYPE_CHECKING:
    from app.modules.annotations.models import AnnotationSet
    from app.modules.auto_label.models import AutoLabelJobItem
    from app.modules.datasets.models import DatasetVersionItem
    from app.modules.projects.models import Project


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        Index("ix_images_project_id", "project_id"),
        Index("ix_images_project_file_hash", "project_id", "file_hash"),
    )

    image_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.project_id"), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String)
    object_key: Mapped[str] = mapped_column(String, nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    project: Mapped["Project"] = relationship(back_populates="images")
    annotation_sets: Mapped[list["AnnotationSet"]] = relationship(back_populates="image")
    dataset_version_items: Mapped[list["DatasetVersionItem"]] = relationship(back_populates="image")
    auto_label_job_items: Mapped[list["AutoLabelJobItem"]] = relationship(back_populates="image")
