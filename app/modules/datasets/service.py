from pathlib import Path
from shutil import make_archive

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.authz import require_project_owner
from app.core.config.settings import get_settings
from app.ml.dataset_builder.yolo import YoloDatasetBuilder
from app.modules.datasets.repository import DatasetRepository
from app.modules.datasets.schemas import DatasetVersionCreate
from app.modules.projects.models import User
from app.modules.training.models import TrainingRun


class DatasetService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = DatasetRepository(db)

    def create_dataset_version(self, data: DatasetVersionCreate, current_user: User):
        require_project_owner(self.db, current_user, data.project_id)

        annotation_sets = self.repository.latest_approved_annotation_sets(data.project_id, data.image_ids)
        if data.image_ids is not None:
            available_ids = {annotation_set.image_id for annotation_set in annotation_sets}
            if set(data.image_ids) != available_ids:
                raise HTTPException(
                    status_code=400,
                    detail="Selected images must be active images in this project with approved labels",
                )
        if not annotation_sets:
            raise HTTPException(status_code=400, detail="No approved annotations with labels found")

        version_number = self.repository.next_version_number(data.project_id)
        dataset_version = self.repository.create_version(
            data, version_number, annotation_sets, current_user.user_id
        )
        self.db.commit()
        return self.repository.get(dataset_version.version_id)

    def list_project_versions(self, project_id: int, current_user: User):
        require_project_owner(self.db, current_user, project_id)
        return self.repository.list_by_project(project_id)

    def delete_dataset_version(self, version_id: int, current_user: User):
        version = self.repository.get(version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Dataset version not found")
        require_project_owner(self.db, current_user, version.project_id)
        run_count = self.db.scalar(
            select(func.count()).select_from(TrainingRun).where(TrainingRun.version_id == version_id)
        ) or 0
        if run_count:
            raise HTTPException(
                status_code=409,
                detail="Dataset version is used by a training run and cannot be deleted",
            )
        self.repository.delete(version_id)
        self.db.commit()
        build_dir = Path(get_settings().temp_root) / "training" / f"dataset_version_{version_id}"
        if build_dir.exists():
            from shutil import rmtree

            rmtree(build_dir)
        archive_path = build_dir.with_suffix(".zip")
        archive_path.unlink(missing_ok=True)

    def build_yolo_dataset(self, version_id: int, current_user: User):
        version = self.repository.get(version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Dataset version not found")
        require_project_owner(self.db, current_user, version.project_id)
        output_dir = YoloDatasetBuilder(self.db).build(version_id)
        summary = self._read_build_summary(output_dir)
        return {
            "version_id": version_id,
            "output_dir": str(output_dir),
            "data_yaml": str(output_dir / "data.yaml"),
            "image_count": summary["image_count"],
            "label_count": summary["label_count"],
        }

    def export_yolo_dataset(self, version_id: int, current_user: User):
        build = self.build_yolo_dataset(version_id, current_user)
        output_dir = Path(build["output_dir"])
        archive_base = output_dir.parent / output_dir.name
        archive_path = Path(make_archive(str(archive_base), "zip", output_dir))
        return {
            "version_id": version_id,
            "archive_path": str(archive_path),
        }

    def get_yolo_export_file_path(self, version_id: int, current_user: User) -> Path:
        export = self.export_yolo_dataset(version_id, current_user)
        path = Path(export["archive_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dataset export not found")
        return path

    def _read_build_summary(self, output_dir):
        values = {"image_count": 0, "label_count": 0}
        summary_path = output_dir / "BUILD_SUMMARY.txt"
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key in values:
                values[key] = int(value)
        return values
