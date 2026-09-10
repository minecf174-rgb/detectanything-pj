from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.session import Base
from app.core.database.types import PrimaryKeyInt

if TYPE_CHECKING:
    from app.modules.datasets.models import DatasetVersion
    from app.modules.models.models import Model
    from app.modules.projects.models import Project


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (
        Index("ix_training_runs_project_id", "project_id"),
        Index("ix_training_runs_version_id", "version_id"),
        Index("ix_training_runs_status", "status"),
    )

    training_run_id: Mapped[int] = mapped_column(
        PrimaryKeyInt, primary_key=True, autoincrement=True
    )
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.project_id"), nullable=False)
    version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.version_id"), nullable=False)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    base_model: Mapped[str | None] = mapped_column(String)
    training_config: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String, nullable=False)
    progress_percent: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    log_object_key: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    project: Mapped["Project"] = relationship(back_populates="training_runs")
    dataset_version: Mapped["DatasetVersion"] = relationship(back_populates="training_runs")
    metrics: Mapped[list["TrainingMetric"]] = relationship(back_populates="training_run")
    model: Mapped["Model | None"] = relationship(back_populates="training_run")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"
    __table_args__ = (
        Index("ix_training_metrics_training_run_id", "training_run_id"),
        Index("ix_training_metrics_training_run_epoch", "training_run_id", "epoch"),
    )

    metric_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int] = mapped_column(
        ForeignKey("training_runs.training_run_id"), nullable=False
    )
    epoch: Mapped[int | None] = mapped_column(Integer)
    box_loss: Mapped[float | None] = mapped_column(Float)
    cls_loss: Mapped[float | None] = mapped_column(Float)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    map50: Mapped[float | None] = mapped_column(Float)
    map50_95: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    training_run: Mapped[TrainingRun] = relationship(back_populates="metrics")
