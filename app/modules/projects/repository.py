from __future__ import annotations

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.modules.annotations.models import Annotation, AnnotationKeypoint, AnnotationSet
from app.modules.auto_label.models import AutoLabelJob, AutoLabelJobItem
from app.modules.classes.models import LabelClass
from app.modules.datasets.models import DatasetVersion, DatasetVersionItem
from app.modules.images.models import Image
from app.modules.models.models import Model, ModelArtifact
from app.modules.projects.models import Project
from app.modules.projects.schemas import ProjectCreate
from app.modules.training.models import TrainingMetric, TrainingRun


class ProjectRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, data: ProjectCreate, owner_id: int) -> Project:
        project = Project(
            owner_id=owner_id,
            name=data.name,
            description=data.description,
            task_type=data.task_type,
        )
        self.db.add(project)
        self.db.flush()
        return project

    def get(self, project_id: int) -> Project | None:
        return self.db.get(Project, project_id)

    def list(self) -> list[Project]:
        return list(self.db.scalars(select(Project).order_by(Project.project_id)))

    def list_by_owner(self, owner_id: int) -> list[Project]:
        return list(
            self.db.scalars(
                select(Project)
                .where(Project.owner_id == owner_id)
                .order_by(Project.project_id)
            )
        )

    def delete_project_tree(self, project_id: int) -> None:
        image_ids = list(self.db.scalars(select(Image.image_id).where(Image.project_id == project_id)))
        version_ids = list(
            self.db.scalars(select(DatasetVersion.version_id).where(DatasetVersion.project_id == project_id))
        )
        run_ids = list(self.db.scalars(select(TrainingRun.training_run_id).where(TrainingRun.project_id == project_id)))
        model_ids = list(self.db.scalars(select(Model.model_id).where(Model.project_id == project_id)))
        job_ids = list(self.db.scalars(select(AutoLabelJob.job_id).where(AutoLabelJob.project_id == project_id)))
        annotation_set_ids = (
            list(self.db.scalars(select(AnnotationSet.annotation_set_id).where(AnnotationSet.image_id.in_(image_ids))))
            if image_ids
            else []
        )

        if job_ids:
            self.db.execute(delete(AutoLabelJobItem).where(AutoLabelJobItem.job_id.in_(job_ids)))
        if image_ids:
            self.db.execute(delete(AutoLabelJobItem).where(AutoLabelJobItem.image_id.in_(image_ids)))
        if job_ids:
            self.db.execute(delete(AutoLabelJob).where(AutoLabelJob.job_id.in_(job_ids)))

        if version_ids:
            self.db.execute(delete(DatasetVersionItem).where(DatasetVersionItem.version_id.in_(version_ids)))
        if image_ids:
            self.db.execute(delete(DatasetVersionItem).where(DatasetVersionItem.image_id.in_(image_ids)))

        if annotation_set_ids:
            annotation_ids = list(
                self.db.scalars(
                    select(Annotation.annotation_id).where(Annotation.annotation_set_id.in_(annotation_set_ids))
                )
            )
            if annotation_ids:
                self.db.execute(delete(AnnotationKeypoint).where(AnnotationKeypoint.annotation_id.in_(annotation_ids)))
            self.db.execute(delete(Annotation).where(Annotation.annotation_set_id.in_(annotation_set_ids)))
            self.db.execute(
                update(AnnotationSet)
                .where(AnnotationSet.parent_annotation_set_id.in_(annotation_set_ids))
                .values(parent_annotation_set_id=None)
            )
            self.db.execute(delete(AnnotationSet).where(AnnotationSet.annotation_set_id.in_(annotation_set_ids)))

        if model_ids:
            self.db.execute(delete(ModelArtifact).where(ModelArtifact.model_id.in_(model_ids)))
        if run_ids:
            self.db.execute(delete(TrainingMetric).where(TrainingMetric.training_run_id.in_(run_ids)))
        if model_ids:
            self.db.execute(delete(Model).where(Model.model_id.in_(model_ids)))
        if run_ids:
            self.db.execute(delete(TrainingRun).where(TrainingRun.training_run_id.in_(run_ids)))
        if version_ids:
            self.db.execute(delete(DatasetVersion).where(DatasetVersion.version_id.in_(version_ids)))
        if image_ids:
            self.db.execute(delete(Image).where(Image.image_id.in_(image_ids)))

        self.db.execute(delete(LabelClass).where(LabelClass.project_id == project_id))
        self.db.execute(delete(Project).where(Project.project_id == project_id))
