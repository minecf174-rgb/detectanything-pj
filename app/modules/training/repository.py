from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.models.models import Model, ModelArtifact
from app.modules.training.models import TrainingMetric, TrainingRun
from app.modules.training.schemas import TrainingRunCreate


class TrainingRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_run(self, data: TrainingRunCreate) -> TrainingRun:
        run = TrainingRun(
            project_id=data.project_id,
            version_id=data.version_id,
            model_type=data.model_type,
            base_model=data.base_model,
            training_config=data.training_config.model_dump() if data.training_config else {},
            status="QUEUED",
            progress_percent=0,
        )
        self.db.add(run)
        self.db.flush()
        return run

    def get_run(self, training_run_id: int) -> TrainingRun | None:
        return self.db.scalar(
            select(TrainingRun)
            .where(TrainingRun.training_run_id == training_run_id)
            .options(selectinload(TrainingRun.metrics))
        )

    def list_by_project(self, project_id: int) -> list[TrainingRun]:
        return list(
            self.db.scalars(
                select(TrainingRun)
                .where(TrainingRun.project_id == project_id)
                .options(selectinload(TrainingRun.metrics))
                .order_by(TrainingRun.training_run_id)
            )
        )

    def next_model_version(self, project_id: int) -> int:
        current = self.db.scalar(select(func.max(Model.version)).where(Model.project_id == project_id))
        return (current or 0) + 1

    def add_metric(self, training_run_id: int, metrics: dict) -> TrainingMetric:
        metric = TrainingMetric(training_run_id=training_run_id, **metrics)
        self.db.add(metric)
        self.db.flush()
        return metric

    def create_model(
        self,
        *,
        project_id: int,
        training_run_id: int,
        version: int,
        weights_object_key: str,
        metrics: dict,
    ) -> Model:
        model = Model(
            project_id=project_id,
            training_run_id=training_run_id,
            source_type="TRAINED",
            model_type=self.db.get(TrainingRun, training_run_id).model_type,
            version=version,
            weights_object_key=weights_object_key,
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            map50=metrics.get("map50"),
            map50_95=metrics.get("map50_95"),
        )
        self.db.add(model)
        self.db.flush()
        self.db.add(
            ModelArtifact(
                model_id=model.model_id,
                artifact_type="BEST_PT",
                object_key=weights_object_key,
            )
        )
        self.db.flush()
        return model
