from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.session import Base
from app.core.database.types import PrimaryKeyInt

if TYPE_CHECKING:
    from app.modules.annotations.models import AnnotationSet
    from app.modules.images.models import Image
    from app.modules.projects.models import Project, User
    from app.modules.training.models import TrainingRun


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("project_id", "version_number", name="uq_dataset_versions_project_version"),
    )

    version_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.project_id"), nullable=False)
    version_number: Mapped[int] = mapped_column(nullable=False)
    preprocess_config: Mapped[dict | None] = mapped_column(JSON)
    augmentation_config: Mapped[dict | None] = mapped_column(JSON)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.user_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped["Project"] = relationship(back_populates="dataset_versions")
    creator: Mapped["User | None"] = relationship()
    items: Mapped[list["DatasetVersionItem"]] = relationship(back_populates="dataset_version")
    training_runs: Mapped[list["TrainingRun"]] = relationship(back_populates="dataset_version")


class DatasetVersionItem(Base):
    __tablename__ = "dataset_version_items"
    __table_args__ = (
        Index("ix_dataset_version_items_annotation_set_id", "annotation_set_id"),
        Index("ix_dataset_version_items_split", "split"),
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"), primary_key=True
    )
    image_id: Mapped[int] = mapped_column(ForeignKey("images.image_id"), primary_key=True)
    annotation_set_id: Mapped[int] = mapped_column(
        ForeignKey("annotation_sets.annotation_set_id"), nullable=False
    )
    split: Mapped[str] = mapped_column(String, nullable=False)

    dataset_version: Mapped[DatasetVersion] = relationship(back_populates="items")
    image: Mapped["Image"] = relationship(back_populates="dataset_version_items")
    annotation_set: Mapped["AnnotationSet"] = relationship()
