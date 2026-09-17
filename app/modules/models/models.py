from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.session import Base
from app.core.database.types import PrimaryKeyInt

if TYPE_CHECKING:
    from app.modules.annotations.models import AnnotationSet
    from app.modules.auto_label.models import AutoLabelJob
    from app.modules.projects.models import Project
    from app.modules.training.models import TrainingRun


class Model(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_models_project_version"),
        Index("ix_models_training_run_id", "training_run_id"),
    )

    model_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.project_id"), nullable=False)
    training_run_id: Mapped[int] = mapped_column(
        ForeignKey("training_runs.training_run_id"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    weights_object_key: Mapped[str] = mapped_column(String, nullable=False)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    map50: Mapped[float | None] = mapped_column(Float)
    map50_95: Mapped[float | None] = mapped_column(Float)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped["Project"] = relationship(back_populates="models")
    training_run: Mapped["TrainingRun"] = relationship(back_populates="model")
    artifacts: Mapped[list["ModelArtifact"]] = relationship(back_populates="model")
    auto_label_jobs: Mapped[list["AutoLabelJob"]] = relationship(back_populates="model")
    annotation_sets: Mapped[list["AnnotationSet"]] = relationship(back_populates="source_model")


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    __table_args__ = (
        UniqueConstraint("model_id", "artifact_type", name="uq_model_artifacts_model_type"),
        Index("ix_model_artifacts_model_id", "model_id"),
    )

    artifact_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.model_id"), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String, nullable=False)
    object_key: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model: Mapped[Model] = relationship(back_populates="artifacts")
