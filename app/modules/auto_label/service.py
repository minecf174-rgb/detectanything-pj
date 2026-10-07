from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.core.authz import require_project_owner
from app.core.config.settings import get_settings
from app.core.database.session import SessionLocal
from app.ml.predictors.ultralytics_yolo import UltralyticsYoloPredictor
from app.modules.annotations.models import AnnotationSet
from app.modules.auto_label.repository import AutoLabelRepository
from app.modules.auto_label.schemas import AutoLabelJobCreate
from app.modules.classes.models import LabelClass
from app.modules.images.models import Image
from app.modules.models.class_mapping import model_class_mapping
from app.modules.models.models import Model
from app.modules.projects.models import Project, User


class AutoLabelService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = AutoLabelRepository(db)
        settings = get_settings()
        self.storage_root = settings.storage_root

    def create_job(self, data: AutoLabelJobCreate, current_user: User):
        require_project_owner(self.db, current_user, data.project_id)
        project = self.db.get(Project, data.project_id)
        if data.base_model is not None:
            if project is None or project.task_type != "POSE_ESTIMATION":
                raise HTTPException(status_code=400, detail="Pretrained pose auto-label is only available for pose projects")
            if not self._pretrained_pose_path().exists():
                raise HTTPException(status_code=404, detail="YOLO26 pose pretrained weights are not installed")
        else:
            model = self.db.get(Model, data.model_id)
            if model is None or model.deleted_at is not None:
                raise HTTPException(status_code=404, detail="Model not found")
            if model.project_id != data.project_id:
                raise HTTPException(status_code=400, detail="Model belongs to another project")
            if model.source_type == "IMPORTED":
                project_classes = list(self.db.scalars(
                    select(LabelClass).where(LabelClass.project_id == data.project_id)
                ))
                model_class_mapping(model, project_classes)

        unique_image_ids = list(dict.fromkeys(data.image_ids))
        images = list(self.db.scalars(select(Image).where(Image.image_id.in_(unique_image_ids))))
        found_ids = {image.image_id for image in images}
        missing_ids = [image_id for image_id in unique_image_ids if image_id not in found_ids]
        if missing_ids:
            raise HTTPException(status_code=404, detail=f"Images not found: {missing_ids}")
        for image in images:
            if image.project_id != data.project_id:
                raise HTTPException(status_code=400, detail="Image belongs to another project")
            if image.status == "ARCHIVED":
                raise HTTPException(status_code=400, detail=f"Image {image.image_id} is archived")

        target_image_ids = unique_image_ids
        if data.only_unlabeled:
            latest_revisions = (
                select(
                    AnnotationSet.image_id,
                    func.max(AnnotationSet.revision).label("revision"),
                )
                .where(AnnotationSet.image_id.in_(unique_image_ids))
                .group_by(AnnotationSet.image_id)
                .subquery()
            )
            unavailable_ids = set(
                self.db.scalars(
                    select(AnnotationSet.image_id)
                    .join(
                        latest_revisions,
                        and_(
                            AnnotationSet.image_id == latest_revisions.c.image_id,
                            AnnotationSet.revision == latest_revisions.c.revision,
                        ),
                    )
                    .where(
                        or_(
                            AnnotationSet.is_negative.is_(True),
                            and_(
                                AnnotationSet.status == "APPROVED",
                                AnnotationSet.annotations.any(),
                            ),
                            and_(
                                AnnotationSet.source_type == "MODEL",
                                AnnotationSet.status == "PENDING",
                            ),
                        ),
                    )
                    .distinct()
                )
            )
            target_image_ids = [image_id for image_id in unique_image_ids if image_id not in unavailable_ids]
            if not target_image_ids:
                raise HTTPException(status_code=400, detail="No unlabeled images available for auto-labeling")

        job = self.repository.create_job(
            project_id=data.project_id,
            model_id=data.model_id,
            base_model=data.base_model,
            image_ids=target_image_ids,
            confidence_threshold=data.confidence_threshold,
            max_predictions_per_image=data.max_predictions_per_image,
        )
        self.db.commit()
        return self.repository.get_job(job.job_id)

    def start_local_mock(self, job_id: int, current_user: User):
        return self._start_local(job_id, predictor_type="mock", current_user=current_user)

    def start_local_yolo(self, job_id: int, current_user: User):
        return self._start_local(job_id, predictor_type="ultralytics", current_user=current_user)

    def start_local_yolo_async(
        self, job_id: int, background_tasks: BackgroundTasks, current_user: User
    ):
        job = self.repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Auto-label job not found")
        require_project_owner(self.db, current_user, job.project_id)
        if job.status not in {"QUEUED", "FAILED"}:
            raise HTTPException(status_code=409, detail=f"Auto-label job is {job.status}")

        self._mark_running(job)
        self.db.commit()
        background_tasks.add_task(run_yolo_auto_label_job, job_id)
        return self.repository.get_job(job_id)

    def _start_local(self, job_id: int, predictor_type: str, current_user: User):
        job = self.repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Auto-label job not found")
        require_project_owner(self.db, current_user, job.project_id)
        if job.status not in {"QUEUED", "FAILED"}:
            raise HTTPException(status_code=409, detail=f"Auto-label job is {job.status}")

        label_class = self.db.scalar(
            select(LabelClass)
            .where(LabelClass.project_id == job.project_id)
            .order_by(LabelClass.class_index)
            .limit(1)
        )
        if label_class is None:
            raise HTTPException(status_code=400, detail="Project has no classes")

        self._mark_running(job)
        self.db.commit()

        predictions = self._predict(job, label_class, predictor_type)
        suggestions: list[AnnotationSet] = []

        for item in job.items:
            boxes = predictions.get(item.image_id, [])
            annotation_set = self.repository.create_model_suggestion(
                image_id=item.image_id,
                model_id=job.model_id,
                boxes=boxes,
            )
            item.annotation_set_id = annotation_set.annotation_set_id
            item.status = "COMPLETED"
            job.processed_images = (job.processed_images or 0) + 1
            suggestions.append(annotation_set)

        job.status = "COMPLETED"
        job.finished_at = datetime.now(timezone.utc)
        self.db.commit()

        refreshed_job = self.repository.get_job(job.job_id)
        refreshed_suggestions = list(
            self.db.scalars(
                select(AnnotationSet)
                .where(AnnotationSet.annotation_set_id.in_([s.annotation_set_id for s in suggestions]))
                .options(selectinload(AnnotationSet.annotations))
                .order_by(AnnotationSet.annotation_set_id)
            )
        )
        return {"job": refreshed_job, "suggestions": refreshed_suggestions}

    def _mark_running(self, job) -> None:
        job.status = "RUNNING"
        job.started_at = datetime.now(timezone.utc)
        job.finished_at = None
        job.error_message = None
        job.processed_images = 0
        for item in job.items:
            item.status = "PENDING"
            item.error_message = None
            item.annotation_set_id = None
        self.db.flush()

    def run_local_yolo_job(self, job_id: int) -> None:
        job = self.repository.get_job(job_id)
        if job is None:
            return
        try:
            label_class = self.db.scalar(
                select(LabelClass)
                .where(LabelClass.project_id == job.project_id)
                .order_by(LabelClass.class_index)
                .limit(1)
            )
            if label_class is None:
                raise RuntimeError("Project has no classes")
            self._run_yolo_predictions_incrementally(job)
            job.status = "COMPLETED"
            job.finished_at = datetime.now(timezone.utc)
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            job = self.repository.get_job(job_id)
            if job is not None:
                job.status = "FAILED"
                job.error_message = str(exc)
                job.finished_at = datetime.now(timezone.utc)
                self.db.commit()

    def _run_yolo_predictions_incrementally(self, job) -> None:
        model_path = self._job_model_path(job)
        if not model_path.exists():
            raise RuntimeError(f"Model weights not found: {model_path}")

        classes = list(
            self.db.scalars(
                select(LabelClass)
                .where(LabelClass.project_id == job.project_id)
                .order_by(LabelClass.class_index)
            )
        )
        source_model = self.db.get(Model, job.model_id) if job.model_id is not None else None
        class_by_index = model_class_mapping(source_model, classes) if source_model else {
            label_class.class_index: label_class for label_class in classes
        }
        predictor = UltralyticsYoloPredictor()
        batch_size = 8
        pending_items = list(job.items)

        for offset in range(0, len(pending_items), batch_size):
            batch_items = pending_items[offset:offset + batch_size]
            image_paths: list[Path] = []
            items_by_path = {}
            for item in batch_items:
                image = self.db.get(Image, item.image_id)
                if image is None:
                    item.status = "FAILED"
                    item.error_message = f"Image not found: {item.image_id}"
                    job.processed_images = (job.processed_images or 0) + 1
                    continue
                path = Path(self.storage_root) / image.object_key
                image_paths.append(path)
                items_by_path[path] = item

            raw_predictions = predictor.predict(
                model_path,
                image_paths,
                {
                    "confidence_threshold": job.confidence_threshold or 0.25,
                    "max_predictions": job.max_predictions_per_image or 20,
                },
            )

            for prediction in raw_predictions:
                item = items_by_path[prediction["image_path"]]
                boxes = []
                for box in prediction["boxes"]:
                    label_class = class_by_index.get(box["class_index"])
                    if label_class is None:
                        continue
                    boxes.append(
                        {
                            "class_id": label_class.class_id,
                            "x_min": box["x_min"],
                            "y_min": box["y_min"],
                            "x_max": box["x_max"],
                            "y_max": box["y_max"],
                            "confidence": box["confidence"],
                            "keypoints": box.get("keypoints", []),
                        }
                    )
                annotation_set = self.repository.create_model_suggestion(
                    image_id=item.image_id,
                    model_id=job.model_id,
                    boxes=boxes,
                )
                item.annotation_set_id = annotation_set.annotation_set_id
                item.status = "COMPLETED"
                job.processed_images = (job.processed_images or 0) + 1
            self.db.commit()

    def _predict(self, job, fallback_class: LabelClass, predictor_type: str) -> dict[int, list[dict]]:
        if predictor_type == "mock":
            confidence = max(float(job.confidence_threshold or 0.25), 0.88)
            return {
                item.image_id: [
                    {
                        "class_id": fallback_class.class_id,
                        "x_min": 0.2,
                        "y_min": 0.2,
                        "x_max": 0.7,
                        "y_max": 0.75,
                        "confidence": confidence,
                    }
                ]
                for item in job.items
            }
        if predictor_type != "ultralytics":
            raise HTTPException(status_code=400, detail=f"Unknown predictor type: {predictor_type}")

        model_path = self._job_model_path(job)

        items_by_path = {}
        image_paths = []
        for item in job.items:
            image = self.db.get(Image, item.image_id)
            if image is None:
                raise HTTPException(status_code=404, detail=f"Image not found: {item.image_id}")
            path = Path(self.storage_root) / image.object_key
            image_paths.append(path)
            items_by_path[path] = item.image_id

        classes = list(
            self.db.scalars(
                select(LabelClass)
                .where(LabelClass.project_id == job.project_id)
                .order_by(LabelClass.class_index)
            )
        )
        source_model = self.db.get(Model, job.model_id) if job.model_id is not None else None
        class_by_index = model_class_mapping(source_model, classes) if source_model else {
            label_class.class_index: label_class for label_class in classes
        }

        raw_predictions = UltralyticsYoloPredictor().predict(
            model_path,
            image_paths,
            {
                "confidence_threshold": job.confidence_threshold or 0.25,
                "max_predictions": job.max_predictions_per_image or 20,
            },
        )
        predictions: dict[int, list[dict]] = {}
        for prediction in raw_predictions:
            image_id = items_by_path[prediction["image_path"]]
            boxes = []
            for box in prediction["boxes"]:
                label_class = class_by_index.get(box["class_index"])
                if label_class is None:
                    continue
                boxes.append(
                    {
                        "class_id": label_class.class_id,
                        "x_min": box["x_min"],
                        "y_min": box["y_min"],
                        "x_max": box["x_max"],
                        "y_max": box["y_max"],
                        "confidence": box["confidence"],
                        "keypoints": box.get("keypoints", []),
                    }
                )
            predictions[image_id] = boxes
        return predictions

    def _job_model_path(self, job) -> Path:
        if job.base_model == "yolo26n-pose.pt":
            return self._pretrained_pose_path()
        model = self.db.get(Model, job.model_id)
        if model is None or model.deleted_at is not None:
            raise HTTPException(status_code=404, detail="Model not found")
        return Path(self.storage_root) / model.weights_object_key

    def _pretrained_pose_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / "yolo26n-pose.pt"

    def list_project_jobs(self, project_id: int, current_user: User):
        require_project_owner(self.db, current_user, project_id)
        return self.repository.list_by_project(project_id)

    def get_job(self, job_id: int, current_user: User):
        job = self.repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Auto-label job not found")
        require_project_owner(self.db, current_user, job.project_id)
        return job


def run_yolo_auto_label_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        AutoLabelService(db).run_local_yolo_job(job_id)
    finally:
        db.close()
