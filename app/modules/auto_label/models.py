from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.session import Base
from app.core.database.types import PrimaryKeyInt

if TYPE_CHECKING:
    from app.modules.annotations.models import AnnotationSet
    from app.modules.images.models import Image
    from app.modules.models.models import Model


class AutoLabelJob(Base):
    __tablename__ = "auto_label_jobs"
    __table_args__ = (
        Index("ix_auto_label_jobs_project_id", "project_id"),
        Index("ix_auto_label_jobs_model_id", "model_id"),
        Index("ix_auto_label_jobs_status", "status"),
    )

    job_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.project_id"), nullable=False)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("models.model_id"))
    base_model: Mapped[str | None] = mapped_column(String)
    confidence_threshold: Mapped[float | None] = mapped_column(Float)
    max_predictions_per_image: Mapped[int | None] = mapped_column(Integer)
    total_images: Mapped[int | None] = mapped_column(Integer)
    processed_images: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    model: Mapped["Model | None"] = relationship(back_populates="auto_label_jobs")
    items: Mapped[list["AutoLabelJobItem"]] = relationship(back_populates="job")


class AutoLabelJobItem(Base):
    __tablename__ = "auto_label_job_items"
    __table_args__ = (
        Index("ix_auto_label_job_items_annotation_set_id", "annotation_set_id"),
        Index("ix_auto_label_job_items_status", "status"),
    )

    job_id: Mapped[int] = mapped_column(ForeignKey("auto_label_jobs.job_id"), primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.image_id"), primary_key=True)
    annotation_set_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotation_sets.annotation_set_id")
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    job: Mapped[AutoLabelJob] = relationship(back_populates="items")
    image: Mapped["Image"] = relationship(back_populates="auto_label_job_items")
    annotation_set: Mapped["AnnotationSet | None"] = relationship()
