from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.authz import require_project_owner
from app.core.config.settings import get_settings
from app.core.storage.service import LocalStorageService
from app.ml.dataset_builder.yolo import YoloDatasetBuilder
from app.ml.trainers.mock import MockYoloTrainer
from app.ml.trainers.ultralytics_yolo import UltralyticsYoloTrainer
from app.modules.classes.models import LabelClass
from app.modules.datasets.models import DatasetVersion
from app.modules.models.class_mapping import model_class_mapping
from app.modules.models.models import Model
from app.modules.projects.models import User
from app.modules.training.models import TrainingRun
from app.modules.training.repository import TrainingRepository
from app.modules.training.schemas import TrainingRunCreate


class TrainingService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = TrainingRepository(db)
        self.storage = LocalStorageService(get_settings().storage_root)

    def create_training_run(self, data: TrainingRunCreate, current_user: User):
        project = require_project_owner(self.db, current_user, data.project_id)
        expected_model_type = (
            "YOLO26_POSE" if project.task_type == "POSE_ESTIMATION" else "YOLO26_DETECT"
        )
        if data.model_type != expected_model_type:
            raise HTTPException(
                status_code=400,
                detail=f"Project task {project.task_type} requires {expected_model_type}",
            )
        if data.base_model and not data.base_model.startswith("model:"):
            is_pose_weights = "-pose.pt" in data.base_model
            if (data.model_type == "YOLO26_POSE") != is_pose_weights:
                raise HTTPException(status_code=400, detail="Base model does not match the project task")
        dataset_version = self.db.get(DatasetVersion, data.version_id)
        if dataset_version is None:
            raise HTTPException(status_code=404, detail="Dataset version not found")
        if dataset_version.project_id != data.project_id:
            raise HTTPException(status_code=400, detail="Dataset version belongs to another project")

        if data.base_model and data.base_model.startswith("model:"):
            source_model_id = int(data.base_model.removeprefix("model:"))
            source_model = self.db.get(Model, source_model_id)
            if source_model is None or source_model.deleted_at is not None:
                raise HTTPException(status_code=404, detail="Source model not found")
            if source_model.project_id != data.project_id:
                raise HTTPException(status_code=400, detail="Source model belongs to another project")
            if source_model.model_type != data.model_type:
                raise HTTPException(status_code=400, detail="Source model task does not match the project")
            if source_model.source_type == "IMPORTED":
                project_classes = list(self.db.scalars(
                    select(LabelClass).where(LabelClass.project_id == data.project_id)
                ))
                mapping = model_class_mapping(source_model, project_classes)
                if any(index != label_class.class_index for index, label_class in mapping.items()):
                    raise HTTPException(status_code=409, detail="Project class indexes no longer match the imported model")
            if not self.storage.exists(source_model.weights_object_key):
                raise HTTPException(status_code=404, detail="Source model weights file not found")

        run = self.repository.create_run(data)
        self.db.commit()
        return self.repository.get_run(run.training_run_id)

    def list_project_runs(self, project_id: int, current_user: User):
        require_project_owner(self.db, current_user, project_id)
        runs = self.repository.list_by_project(project_id)
        for run in runs:
            run.can_resume = self._checkpoint_path(run).exists() and run.status in {
                "FAILED", "INTERRUPTED"
            }
        return runs

    def start_local_mock(self, training_run_id: int, current_user: User):
        return self._start_local(training_run_id, trainer_type="mock", current_user=current_user)

    def start_local_yolo(self, training_run_id: int, current_user: User):
        return self._start_local(training_run_id, trainer_type="ultralytics", current_user=current_user)

    def resume_local_yolo(self, training_run_id: int, current_user: User):
        run = self.repository.get_run(training_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Training run not found")
        require_project_owner(self.db, current_user, run.project_id)
        if run.status not in {"FAILED", "INTERRUPTED"}:
            raise HTTPException(status_code=409, detail=f"Training run is {run.status}")
        checkpoint_path = self._checkpoint_path(run)
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail="Training checkpoint not found")

        run.status = "RUNNING"
        run.error_message = None
        run.finished_at = None
        self.db.commit()

        try:
            dataset_dir = YoloDatasetBuilder(self.db).build(run.version_id)
            result = UltralyticsYoloTrainer().resume(checkpoint_path)
            return self._complete_run(run, result, dataset_dir)
        except Exception as exc:
            self._mark_failed(training_run_id, exc)
            raise

    def _start_local(self, training_run_id: int, trainer_type: str, current_user: User):
        run = self.repository.get_run(training_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Training run not found")
        require_project_owner(self.db, current_user, run.project_id)
        if run.status not in {"QUEUED", "FAILED"}:
            raise HTTPException(status_code=409, detail=f"Training run is {run.status}")

        run.status = "RUNNING"
        run.progress_percent = 10
        run.started_at = datetime.now(timezone.utc)
        run.error_message = None
        run.finished_at = None
        self.db.commit()

        try:
            dataset_dir = YoloDatasetBuilder(self.db).build(run.version_id)
            result = self._run_trainer(run, dataset_dir, trainer_type)
            artifact_name = "best_mock.pt" if trainer_type == "mock" else "best.pt"
            return self._complete_run(run, result, dataset_dir, artifact_name)
        except IntegrityError as exc:
            self._mark_failed(training_run_id, exc)
            raise HTTPException(status_code=409, detail="Model version already exists") from exc
        except Exception as exc:
            self._mark_failed(training_run_id, exc)
            raise

    def _complete_run(self, run, result: dict, dataset_dir: Path, artifact_name: str = "best.pt"):
        metrics = result["metrics"]
        self.repository.add_metric(run.training_run_id, metrics)
        model_version = self.repository.next_model_version(run.project_id)
        weights_object_key = f"models/project_{run.project_id}/model_{model_version}/{artifact_name}"
        self._save_weights(weights_object_key, result)
        model = self.repository.create_model(
            project_id=run.project_id,
            training_run_id=run.training_run_id,
            version=model_version,
            weights_object_key=weights_object_key,
            metrics=metrics,
        )
        run.status = "COMPLETED"
        run.progress_percent = 100
        run.error_message = None
        run.finished_at = datetime.now(timezone.utc)
        self.db.commit()
        response = {
            "training_run": self.repository.get_run(run.training_run_id),
            "model": model,
            "dataset_dir": str(dataset_dir),
        }
        self._cleanup_completed_run(run)
        return response

    def _mark_failed(self, training_run_id: int, exc: Exception) -> None:
        self.db.rollback()
        failed_run = self.repository.get_run(training_run_id)
        if failed_run is None:
            return
        failed_run.status = "FAILED"
        failed_run.error_message = str(exc)
        failed_run.finished_at = datetime.now(timezone.utc)
        self.db.commit()

    def _checkpoint_path(self, run) -> Path:
        return (
            Path(get_settings().temp_root)
            / "training"
            / "runs"
            / f"training_run_{run.training_run_id}"
            / "weights"
            / "last.pt"
        )

    def _cleanup_completed_run(self, run) -> None:
        run_dir = self._checkpoint_path(run).parents[1]
        rmtree(run_dir, ignore_errors=True)

        another_run_is_active = self.db.query(TrainingRun).filter(
            TrainingRun.version_id == run.version_id,
            TrainingRun.training_run_id != run.training_run_id,
            TrainingRun.status == "RUNNING",
        ).first() is not None
        if another_run_is_active:
            return

        dataset_dir = (
            Path(get_settings().temp_root)
            / "training"
            / f"dataset_version_{run.version_id}"
        )
        rmtree(dataset_dir, ignore_errors=True)
        dataset_dir.with_suffix(".zip").unlink(missing_ok=True)

    @classmethod
    def recover_interrupted_runs(cls, db: Session) -> int:
        service = cls(db)
        recovered = 0
        runs = db.query(TrainingRun).filter(TrainingRun.status.in_({"RUNNING", "QUEUED"})).all()
        for run in runs:
            if run.status == "QUEUED" and not service._checkpoint_path(run).exists():
                continue
            run.status = "INTERRUPTED" if service._checkpoint_path(run).exists() else "FAILED"
            run.error_message = (
                "Training was interrupted. Resume from the latest checkpoint."
                if run.status == "INTERRUPTED"
                else "Training stopped before a resumable checkpoint was written."
            )
            run.finished_at = datetime.now(timezone.utc)
            recovered += 1
        if recovered:
            db.commit()
        for run in db.query(TrainingRun).filter_by(status="COMPLETED").all():
            service._cleanup_completed_run(run)
        return recovered

    def _run_trainer(self, run, dataset_dir, trainer_type: str):
        config = dict(run.training_config or {})
        if trainer_type == "mock":
            return MockYoloTrainer().train(dataset_dir, config)
        if trainer_type == "ultralytics":
            config.setdefault("base_model", self._resolve_base_model(run))
            config.setdefault("epochs", 50)
            config.setdefault("imgsz", 640)
            config.setdefault("batch", 4)
            config.setdefault("run_name", f"training_run_{run.training_run_id}")
            return UltralyticsYoloTrainer().train(dataset_dir, config)
        raise HTTPException(status_code=400, detail=f"Unknown trainer type: {trainer_type}")

    def _resolve_base_model(self, run) -> str:
        base_model = run.base_model or "yolo26s.pt"
        if not base_model.startswith("model:"):
            return base_model
        source_model_id = int(base_model.removeprefix("model:"))
        source_model = self.db.get(Model, source_model_id)
        if source_model is None or source_model.deleted_at is not None or source_model.project_id != run.project_id:
            raise HTTPException(status_code=400, detail="Invalid source model")
        source_path = self.storage.root / source_model.weights_object_key
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source model weights file not found")
        return str(source_path.resolve())

    def _save_weights(self, object_key: str, result: dict):
        if "weights" in result:
            self.storage.save(object_key, result["weights"])
            return

        weights_path = result.get("weights_path")
        if weights_path is None:
            raise HTTPException(status_code=500, detail="Trainer did not return weights")
        target = self.storage.root / object_key
        UltralyticsYoloTrainer().copy_weights_to(weights_path, target)
