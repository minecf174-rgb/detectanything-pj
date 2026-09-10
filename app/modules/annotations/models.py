from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.session import Base
from app.core.database.types import PrimaryKeyInt

if TYPE_CHECKING:
    from app.modules.classes.models import LabelClass
    from app.modules.images.models import Image
    from app.modules.models.models import Model
    from app.modules.projects.models import User


class AnnotationSet(Base):
    __tablename__ = "annotation_sets"
    __table_args__ = (
        UniqueConstraint("image_id", "revision", name="uq_annotation_sets_image_revision"),
        Index("ix_annotation_sets_image_id", "image_id"),
        Index("ix_annotation_sets_source_model_id", "source_model_id"),
        Index("ix_annotation_sets_parent_annotation_set_id", "parent_annotation_set_id"),
    )

    annotation_set_id: Mapped[int] = mapped_column(
        PrimaryKeyInt, primary_key=True, autoincrement=True
    )
    image_id: Mapped[int] = mapped_column(ForeignKey("images.image_id"), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_model_id: Mapped[int | None] = mapped_column(ForeignKey("models.model_id"))
    parent_annotation_set_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotation_sets.annotation_set_id")
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.user_id"))
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.user_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    image: Mapped["Image"] = relationship(back_populates="annotation_sets")
    source_model: Mapped["Model | None"] = relationship(back_populates="annotation_sets")
    parent_annotation_set: Mapped["AnnotationSet | None"] = relationship(
        remote_side=[annotation_set_id]
    )
    creator: Mapped["User | None"] = relationship(foreign_keys=[created_by])
    reviewer: Mapped["User | None"] = relationship(foreign_keys=[reviewed_by])
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="annotation_set")


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        Index("ix_annotations_annotation_set_id", "annotation_set_id"),
        Index("ix_annotations_class_id", "class_id"),
    )

    annotation_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    annotation_set_id: Mapped[int] = mapped_column(
        ForeignKey("annotation_sets.annotation_set_id"), nullable=False
    )
    class_id: Mapped[int] = mapped_column(ForeignKey("classes.class_id"), nullable=False)
    x_min: Mapped[float] = mapped_column(Float, nullable=False)
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    annotation_set: Mapped[AnnotationSet] = relationship(back_populates="annotations")
    label_class: Mapped["LabelClass"] = relationship(back_populates="annotations")
    keypoints: Mapped[list["AnnotationKeypoint"]] = relationship(back_populates="annotation")


class AnnotationKeypoint(Base):
    __tablename__ = "annotation_keypoints"
    __table_args__ = (
        Index("ix_annotation_keypoints_annotation_id", "annotation_id"),
        UniqueConstraint("annotation_id", "keypoint_index", name="uq_annotation_keypoints_index"),
    )

    keypoint_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    annotation_id: Mapped[int] = mapped_column(ForeignKey("annotations.annotation_id"), nullable=False)
    keypoint_index: Mapped[int] = mapped_column(Integer, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    visibility: Mapped[int] = mapped_column(Integer, nullable=False, default=2)

    annotation: Mapped[Annotation] = relationship(back_populates="keypoints")
