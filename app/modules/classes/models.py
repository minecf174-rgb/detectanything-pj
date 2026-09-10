from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.session import Base
from app.core.database.types import PrimaryKeyInt

if TYPE_CHECKING:
    from app.modules.annotations.models import Annotation
    from app.modules.projects.models import Project


class LabelClass(Base):
    __tablename__ = "classes"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_classes_project_name"),
        UniqueConstraint("project_id", "class_index", name="uq_classes_project_class_index"),
        Index("ix_classes_project_id", "project_id"),
    )

    class_id: Mapped[int] = mapped_column(PrimaryKeyInt, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.project_id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    class_index: Mapped[int] = mapped_column(Integer, nullable=False)
    color: Mapped[str] = mapped_column(String(16), nullable=False, default="#16a34a", server_default="#16a34a")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped["Project"] = relationship(back_populates="classes")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="label_class")
