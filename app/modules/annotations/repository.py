from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.annotations.models import Annotation, AnnotationKeypoint, AnnotationSet
from app.modules.annotations.schemas import AnnotationCorrectionRequest, AnnotationSetCreate


class AnnotationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def next_revision(self, image_id: int) -> int:
        current = self.db.scalar(
            select(func.max(AnnotationSet.revision)).where(AnnotationSet.image_id == image_id)
        )
        return (current or 0) + 1

    def create_set(
        self, image_id: int, revision: int, data: AnnotationSetCreate, created_by: int
    ) -> AnnotationSet:
        annotation_set = AnnotationSet(
            image_id=image_id,
            revision=revision,
            source_type="HUMAN",
            status="APPROVED",
            created_by=created_by,
        )
        self.db.add(annotation_set)
        self.db.flush()

        annotations = [
            Annotation(
                    annotation_set_id=annotation_set.annotation_set_id,
                    class_id=item.class_id,
                    x_min=item.x_min,
                    y_min=item.y_min,
                    x_max=item.x_max,
                    y_max=item.y_max,
                    confidence=item.confidence,
                )
            for item in data.annotations
        ]
        self.db.add_all(annotations)
        self.db.flush()
        self._add_keypoints(annotations, data.annotations)
        return annotation_set

    def list_by_image(self, image_id: int) -> list[AnnotationSet]:
        return list(
            self.db.scalars(
                select(AnnotationSet)
                .where(AnnotationSet.image_id == image_id)
                .options(selectinload(AnnotationSet.annotations).selectinload(Annotation.keypoints))
                .order_by(AnnotationSet.revision)
            )
        )

    def get_set(self, annotation_set_id: int) -> AnnotationSet | None:
        return self.db.scalar(
            select(AnnotationSet)
            .where(AnnotationSet.annotation_set_id == annotation_set_id)
            .options(selectinload(AnnotationSet.annotations).selectinload(Annotation.keypoints))
        )

    def create_correction(
        self,
        source: AnnotationSet,
        revision: int,
        data: AnnotationCorrectionRequest,
        reviewed_at,
        reviewed_by: int,
    ) -> AnnotationSet:
        corrected_set = AnnotationSet(
            image_id=source.image_id,
            revision=revision,
            source_type="HUMAN_CORRECTED",
            source_model_id=source.source_model_id,
            parent_annotation_set_id=source.annotation_set_id,
            status="APPROVED",
            is_negative=source.is_negative and not data.annotations,
            created_by=reviewed_by,
            reviewed_by=reviewed_by,
            reviewed_at=reviewed_at,
        )
        self.db.add(corrected_set)
        self.db.flush()

        annotations = [
            Annotation(
                    annotation_set_id=corrected_set.annotation_set_id,
                    class_id=item.class_id,
                    x_min=item.x_min,
                    y_min=item.y_min,
                    x_max=item.x_max,
                    y_max=item.y_max,
                    confidence=None,
                )
            for item in data.annotations
        ]
        self.db.add_all(annotations)
        self.db.flush()
        self._add_keypoints(annotations, data.annotations)
        return corrected_set

    def _add_keypoints(self, annotations, items) -> None:
        keypoints = [
            AnnotationKeypoint(
                annotation_id=annotation.annotation_id,
                keypoint_index=index,
                x=keypoint.x,
                y=keypoint.y,
                visibility=keypoint.visibility,
            )
            for annotation, item in zip(annotations, items, strict=True)
            for index, keypoint in enumerate(item.keypoints)
        ]
        if keypoints:
            self.db.add_all(keypoints)
            self.db.flush()
