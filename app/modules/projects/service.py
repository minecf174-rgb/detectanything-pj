from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config.settings import get_settings
from app.core.authz import require_project_owner
from app.core.storage.service import LocalStorageService
from app.modules.auto_label.models import AutoLabelJob
from app.modules.datasets.models import DatasetVersion
from app.modules.projects.models import User
from app.modules.projects.repository import ProjectRepository
from app.modules.projects.schemas import ProjectCreate
from app.modules.training.models import TrainingRun


class ProjectService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = ProjectRepository(db)
        settings = get_settings()
        self.storage = LocalStorageService(settings.storage_root)
        self.temp_storage = LocalStorageService(settings.temp_root)

    def create_project(self, data: ProjectCreate, current_user: User):
        project = self.repository.create(data, current_user.user_id)
        self.db.commit()
        self.db.refresh(project)
        return project

    def list_projects(self, current_user: User, owner_id: int | None = None):
        if owner_id is not None and owner_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="You can only list your own projects")
        return self.repository.list_by_owner(current_user.user_id)

    def get_project(self, project_id: int, current_user: User):
        return require_project_owner(self.db, current_user, project_id)

    def delete_project(self, project_id: int, current_user: User) -> None:
        require_project_owner(self.db, current_user, project_id)
        self.ensure_no_running_jobs(project_id)
        version_ids, run_ids = self.artifact_ids(project_id)
        self.repository.delete_project_tree(project_id)
        self.db.commit()
        self.cleanup_files(project_id, version_ids, run_ids)

    def ensure_no_running_jobs(self, project_id: int) -> None:
        active_training = self.db.scalar(
            select(TrainingRun.training_run_id)
            .where(TrainingRun.project_id == project_id, TrainingRun.status.in_(("RUNNING", "QUEUED")))
            .limit(1)
        )
        active_auto_label = self.db.scalar(
            select(AutoLabelJob.job_id)
            .where(AutoLabelJob.project_id == project_id, AutoLabelJob.status.in_(("RUNNING", "QUEUED")))
            .limit(1)
        )
        if active_training is not None or active_auto_label is not None:
            raise HTTPException(
                status_code=409,
                detail="Wait for training and auto-label jobs to finish before deleting this project",
            )

    def artifact_ids(self, project_id: int) -> tuple[list[int], list[int]]:
        version_ids = list(self.db.scalars(
            select(DatasetVersion.version_id).where(DatasetVersion.project_id == project_id)
        ))
        run_ids = list(self.db.scalars(
            select(TrainingRun.training_run_id).where(TrainingRun.project_id == project_id)
        ))
        return version_ids, run_ids

    def cleanup_files(self, project_id: int, version_ids: list[int], run_ids: list[int]) -> None:
        self.storage.delete_prefix(f"projects/project_{project_id}")
        self.storage.delete_prefix(f"previews/project_{project_id}")
        self.storage.delete_prefix(f"models/project_{project_id}")
        for version_id in version_ids:
            self.temp_storage.delete_prefix(f"training/dataset_version_{version_id}")
            self.temp_storage.delete_prefix(f"training/dataset_version_{version_id}.zip")
        for run_id in run_ids:
            self.temp_storage.delete_prefix(f"training/runs/training_run_{run_id}")
