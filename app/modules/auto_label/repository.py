from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app.modules.annotations.models import Annotation, AnnotationKeypoint, AnnotationSet
from app.modules.auto_label.models import AutoLabelJob, AutoLabelJobItem


class AutoLabelRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_job(
        self,
        *,
        project_id: int,
        model_id: int | None,
        base_model: str | None,
        image_ids: list[int],
        confidence_threshold: float,
        max_predictions_per_image: int | None,
    ) -> AutoLabelJob:
        job = AutoLabelJob(
            project_id=project_id,
            model_id=model_id,
            base_model=base_model,
            confidence_threshold=confidence_threshold,
            max_predictions_per_image=max_predictions_per_image,
            total_images=len(image_ids),
            processed_images=0,
            status="QUEUED",
        )
        self.db.add(job)
        self.db.flush()
        self.db.add_all(
            [AutoLabelJobItem(job_id=job.job_id, image_id=image_id, status="PENDING") for image_id in image_ids]
        )
        self.db.flush()
        return job

    def get_job(self, job_id: int) -> AutoLabelJob | None:
        return self.db.scalar(
            select(AutoLabelJob)
            .where(AutoLabelJob.job_id == job_id)
            .options(selectinload(AutoLabelJob.items))
        )

    def list_by_project(self, project_id: int) -> list[AutoLabelJob]:
        return list(
            self.db.scalars(
                select(AutoLabelJob)
                .where(AutoLabelJob.project_id == project_id)
                .options(selectinload(AutoLabelJob.items))
                .order_by(AutoLabelJob.job_id)
            )
        )

    def next_revision(self, image_id: int) -> int:
        current = self.db.scalar(
            select(func.max(AnnotationSet.revision)).where(AnnotationSet.image_id == image_id)
        )
        return (current or 0) + 1

    def create_model_suggestion(
        self,
        *,
        image_id: int,
        model_id: int | None,
        boxes: list[dict],
    ) -> AnnotationSet:
        annotation_set = AnnotationSet(
            image_id=image_id,
            revision=self.next_revision(image_id),
            source_type="MODEL",
            source_model_id=model_id,
            status="PENDING",
        )
        self.db.add(annotation_set)
        self.db.flush()
        annotations = [
            Annotation(
                    annotation_set_id=annotation_set.annotation_set_id,
                    class_id=box["class_id"],
                    x_min=box["x_min"],
                    y_min=box["y_min"],
                    x_max=box["x_max"],
                    y_max=box["y_max"],
                    confidence=box["confidence"],
                )
            for box in boxes
        ]
        self.db.add_all(annotations)
        self.db.flush()
        keypoints = [
            AnnotationKeypoint(
                annotation_id=annotation.annotation_id,
                keypoint_index=index,
                x=keypoint["x"],
                y=keypoint["y"],
                visibility=keypoint.get("visibility", 2),
            )
            for annotation, box in zip(annotations, boxes, strict=True)
            for index, keypoint in enumerate(box.get("keypoints", []))
        ]
        if keypoints:
            self.db.add_all(keypoints)
            self.db.flush()
        return annotation_set

    def mark_item_reviewed_by_annotation_set(self, annotation_set_id: int) -> None:
        item = self.db.scalar(
            select(AutoLabelJobItem).where(
                AutoLabelJobItem.annotation_set_id == annotation_set_id
            )
        )
        if item is not None:
            item.status = "REVIEWED"

    def mark_items_reviewed_by_annotation_sets(self, annotation_set_ids: list[int]) -> None:
        if not annotation_set_ids:
            return
        self.db.execute(
            update(AutoLabelJobItem)
            .where(AutoLabelJobItem.annotation_set_id.in_(annotation_set_ids))
            .values(status="REVIEWED")
        )
