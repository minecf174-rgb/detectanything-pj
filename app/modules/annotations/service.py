from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.annotations.models import Annotation, AnnotationSet
from app.core.authz import require_project_owner
from app.modules.annotations.repository import AnnotationRepository
from datetime import datetime, timezone

from app.modules.annotations.schemas import (
    AnnotationBulkReviewRequest,
    AnnotationCorrectionRequest,
    AnnotationPreviewBatchRequest,
    AnnotationReviewRequest,
    AnnotationSetCreate,
)
from app.modules.auto_label.repository import AutoLabelRepository
from app.modules.classes.models import LabelClass
from app.modules.images.models import Image
from app.modules.projects.models import Project, User


class AnnotationService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = AnnotationRepository(db)

    def create_annotation_set(self, image_id: int, data: AnnotationSetCreate, current_user: User):
        image = self.db.get(Image, image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        require_project_owner(self.db, current_user, image.project_id)

        for item in data.annotations:
            label_class = self.db.get(LabelClass, item.class_id)
            if label_class is None:
                raise HTTPException(status_code=404, detail=f"Class {item.class_id} not found")
            if label_class.project_id != image.project_id:
                raise HTTPException(status_code=400, detail="Class and image belong to different projects")
        self._validate_project_annotation_shape(image.project_id, data.annotations)

        revision = self.repository.next_revision(image_id)
        annotation_set = self.repository.create_set(image_id, revision, data, current_user.user_id)
        self.db.commit()
        self.db.refresh(annotation_set)
        return self.repository.list_by_image(image_id)[-1]

    def list_image_annotation_sets(self, image_id: int, current_user: User):
        image = self.db.get(Image, image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        require_project_owner(self.db, current_user, image.project_id)
        return self.repository.list_by_image(image_id)

    def list_annotation_previews(self, data: AnnotationPreviewBatchRequest, current_user: User):
        annotation_set_ids = list(dict.fromkeys(data.annotation_set_ids))
        previews = list(self.db.scalars(
            select(AnnotationSet)
            .join(Image, Image.image_id == AnnotationSet.image_id)
            .join(Project, Project.project_id == Image.project_id)
            .where(
                AnnotationSet.annotation_set_id.in_(annotation_set_ids),
                Project.owner_id == current_user.user_id,
                Image.status != "ARCHIVED",
            )
            .options(selectinload(AnnotationSet.annotations).selectinload(Annotation.keypoints))
        ))
        if len(previews) != len(annotation_set_ids):
            raise HTTPException(status_code=404, detail="Annotation preview not found")
        return previews

    def list_project_image_statuses(self, project_id: int, current_user: User):
        require_project_owner(self.db, current_user, project_id)
        rows = self.db.execute(
            select(
                AnnotationSet.image_id,
                AnnotationSet.annotation_set_id,
                AnnotationSet.source_type,
                AnnotationSet.status,
                AnnotationSet.is_negative,
                AnnotationSet.revision,
                func.count(Annotation.annotation_id).label("box_count"),
            )
            .join(Image, Image.image_id == AnnotationSet.image_id)
            .outerjoin(Annotation, Annotation.annotation_set_id == AnnotationSet.annotation_set_id)
            .where(Image.project_id == project_id, Image.status != "ARCHIVED")
            .group_by(AnnotationSet.annotation_set_id)
            .order_by(AnnotationSet.image_id, AnnotationSet.revision.desc())
        ).all()
        latest_by_image: dict[int, dict] = {}
        latest_approved_by_image: dict[int, int] = {}
        pending_set_ids: set[int] = set()
        for image_id, annotation_set_id, source_type, status, is_negative, _revision, box_count in rows:
            if image_id not in latest_by_image:
                latest_by_image[image_id] = {
                    "image_id": image_id,
                    "annotation_set_id": annotation_set_id,
                    "source_type": source_type,
                    "status": status,
                    "box_count": box_count,
                    "is_negative": is_negative,
                    "class_ids": [],
                }
            if status == "APPROVED" and image_id not in latest_approved_by_image:
                latest_approved_by_image[image_id] = annotation_set_id
            elif status == "PENDING":
                pending_set_ids.add(annotation_set_id)
        active_set_ids = set(latest_approved_by_image.values()) | pending_set_ids
        if active_set_ids:
            set_image_ids = {annotation_set_id: image_id for image_id, annotation_set_id, *_ in rows
                             if annotation_set_id in active_set_ids}
            class_ids_by_image: dict[int, set[int]] = {}
            active_ids = sorted(active_set_ids)
            for offset in range(0, len(active_ids), 500):
                for annotation_set_id, class_id in self.db.execute(
                    select(Annotation.annotation_set_id, Annotation.class_id)
                    .where(Annotation.annotation_set_id.in_(active_ids[offset:offset + 500]))
                ):
                    image_id = set_image_ids[annotation_set_id]
                    class_ids_by_image.setdefault(image_id, set()).add(class_id)
            for image_id, class_ids in class_ids_by_image.items():
                latest_by_image[image_id]["class_ids"] = sorted(class_ids)
        return list(latest_by_image.values())

    def confirm_negative_image(self, image_id: int, current_user: User):
        image = self.db.get(Image, image_id)
        if image is None or image.status == "ARCHIVED":
            raise HTTPException(status_code=404, detail="Active image not found")
        require_project_owner(self.db, current_user, image.project_id)
        latest = self.db.scalar(
            select(AnnotationSet)
            .where(AnnotationSet.image_id == image_id)
            .order_by(AnnotationSet.revision.desc())
            .limit(1)
        )
        pending_sets = list(self.db.scalars(
            select(AnnotationSet).where(
                AnnotationSet.image_id == image_id,
                AnnotationSet.source_type == "MODEL",
                AnnotationSet.status == "PENDING",
            )
        ))
        reviewed_at = datetime.now(timezone.utc)
        for annotation_set in pending_sets:
            annotation_set.status = "REJECTED"
            annotation_set.reviewed_by = current_user.user_id
            annotation_set.reviewed_at = reviewed_at
        AutoLabelRepository(self.db).mark_items_reviewed_by_annotation_sets(
            [annotation_set.annotation_set_id for annotation_set in pending_sets]
        )
        if latest is not None and latest.is_negative:
            self.db.commit()
            return self.repository.get_set(latest.annotation_set_id)
        negative = self.repository.create_set(
            image_id, self.repository.next_revision(image_id), AnnotationSetCreate(annotations=[]), current_user.user_id
        )
        negative.is_negative = True
        negative.reviewed_by = current_user.user_id
        negative.reviewed_at = reviewed_at
        self.db.commit()
        return self.repository.get_set(negative.annotation_set_id)

    def accept_annotation_set(
        self, annotation_set_id: int, data: AnnotationReviewRequest, current_user: User
    ):
        annotation_set = self.repository.get_set(annotation_set_id)
        if annotation_set is None:
            raise HTTPException(status_code=404, detail="Annotation set not found")
        self._require_annotation_set_owner(annotation_set, current_user)
        if annotation_set.source_type != "MODEL":
            raise HTTPException(status_code=400, detail="Only MODEL annotation sets can be reviewed")
        # Retrying after a lost response must not turn a successful accept into a conflict.
        if annotation_set.status == "APPROVED":
            return annotation_set
        if annotation_set.status != "PENDING":
            raise HTTPException(status_code=409, detail=f"Annotation set is {annotation_set.status}")
        annotation_set.status = "APPROVED"
        annotation_set.reviewed_by = current_user.user_id
        annotation_set.reviewed_at = datetime.now(timezone.utc)
        AutoLabelRepository(self.db).mark_item_reviewed_by_annotation_set(annotation_set_id)
        self.db.commit()
        return self.repository.get_set(annotation_set_id)

    def reject_annotation_set(
        self, annotation_set_id: int, data: AnnotationReviewRequest, current_user: User
    ):
        annotation_set = self._get_pending_model_annotation_set(annotation_set_id)
        self._require_annotation_set_owner(annotation_set, current_user)
        annotation_set.status = "REJECTED"
        annotation_set.reviewed_by = current_user.user_id
        annotation_set.reviewed_at = datetime.now(timezone.utc)
        AutoLabelRepository(self.db).mark_item_reviewed_by_annotation_set(annotation_set_id)
        self.db.commit()
        return self.repository.get_set(annotation_set_id)

    def reject_annotation_sets_bulk(self, data: AnnotationBulkReviewRequest, current_user: User):
        requested_ids = set(data.annotation_set_ids)
        pending_sets = list(
            self.db.scalars(
                select(AnnotationSet).where(
                    AnnotationSet.annotation_set_id.in_(requested_ids),
                    AnnotationSet.source_type == "MODEL",
                    AnnotationSet.status == "PENDING",
                )
            )
        )
        reviewed_at = datetime.now(timezone.utc)
        rejected_ids: list[int] = []
        for annotation_set in pending_sets:
            self._require_annotation_set_owner(annotation_set, current_user)
            annotation_set.status = "REJECTED"
            annotation_set.reviewed_by = current_user.user_id
            annotation_set.reviewed_at = reviewed_at
            rejected_ids.append(annotation_set.annotation_set_id)
        AutoLabelRepository(self.db).mark_items_reviewed_by_annotation_sets(rejected_ids)
        self.db.commit()
        return {
            "rejected_count": len(rejected_ids),
            "skipped_count": len(requested_ids) - len(rejected_ids),
        }

    def accept_annotation_sets_bulk(self, data: AnnotationBulkReviewRequest, current_user: User):
        requested_ids = set(data.annotation_set_ids)
        pending_sets = list(
            self.db.scalars(
                select(AnnotationSet).where(
                    AnnotationSet.annotation_set_id.in_(requested_ids),
                    AnnotationSet.source_type == "MODEL",
                    AnnotationSet.status == "PENDING",
                )
            )
        )
        reviewed_at = datetime.now(timezone.utc)
        accepted_ids: list[int] = []
        for annotation_set in pending_sets:
            self._require_annotation_set_owner(annotation_set, current_user)
            annotation_set.status = "APPROVED"
            annotation_set.reviewed_by = current_user.user_id
            annotation_set.reviewed_at = reviewed_at
            accepted_ids.append(annotation_set.annotation_set_id)
        AutoLabelRepository(self.db).mark_items_reviewed_by_annotation_sets(accepted_ids)
        self.db.commit()
        return {
            "accepted_count": len(accepted_ids),
            "skipped_count": len(requested_ids) - len(accepted_ids),
        }

    def correct_annotation_set(
        self, annotation_set_id: int, data: AnnotationCorrectionRequest, current_user: User
    ):
        source = self._get_pending_model_annotation_set(annotation_set_id)
        self._require_annotation_set_owner(source, current_user)
        image = self.db.get(Image, source.image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        for item in data.annotations:
            label_class = self.db.get(LabelClass, item.class_id)
            if label_class is None:
                raise HTTPException(status_code=404, detail=f"Class {item.class_id} not found")
            if label_class.project_id != image.project_id:
                raise HTTPException(status_code=400, detail="Class and image belong to different projects")

        self._validate_project_annotation_shape(image.project_id, data.annotations)

        reviewed_at = datetime.now(timezone.utc)
        source.status = "REJECTED"
        source.reviewed_by = current_user.user_id
        source.reviewed_at = reviewed_at
        AutoLabelRepository(self.db).mark_item_reviewed_by_annotation_set(annotation_set_id)

        revision = self.repository.next_revision(source.image_id)
        corrected = self.repository.create_correction(
            source, revision, data, reviewed_at, current_user.user_id
        )
        self.db.commit()
        return self.repository.get_set(corrected.annotation_set_id)

    def edit_approved_annotation_set(
        self, annotation_set_id: int, data: AnnotationCorrectionRequest, current_user: User
    ):
        source = self.repository.get_set(annotation_set_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Annotation set not found")
        if source.status != "APPROVED":
            raise HTTPException(status_code=409, detail=f"Annotation set is {source.status}")
        self._require_annotation_set_owner(source, current_user)
        image = self.db.get(Image, source.image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        for item in data.annotations:
            label_class = self.db.get(LabelClass, item.class_id)
            if label_class is None:
                raise HTTPException(status_code=404, detail=f"Class {item.class_id} not found")
            if label_class.project_id != image.project_id:
                raise HTTPException(status_code=400, detail="Class and image belong to different projects")

        self._validate_project_annotation_shape(image.project_id, data.annotations)

        corrected = self.repository.create_correction(
            source,
            self.repository.next_revision(source.image_id),
            data,
            datetime.now(timezone.utc),
            current_user.user_id,
        )
        self.db.commit()
        return self.repository.get_set(corrected.annotation_set_id)

    def _get_pending_model_annotation_set(self, annotation_set_id: int):
        annotation_set = self.repository.get_set(annotation_set_id)
        if annotation_set is None:
            raise HTTPException(status_code=404, detail="Annotation set not found")
        if annotation_set.source_type != "MODEL":
            raise HTTPException(status_code=400, detail="Only MODEL annotation sets can be reviewed")
        if annotation_set.status != "PENDING":
            raise HTTPException(status_code=409, detail=f"Annotation set is {annotation_set.status}")
        return annotation_set

    def _validate_user(self, user_id: int) -> None:
        if self.db.get(User, user_id) is None:
            raise HTTPException(status_code=404, detail="Reviewer user not found")

    def _require_annotation_set_owner(self, annotation_set: AnnotationSet, current_user: User) -> None:
        image = self.db.get(Image, annotation_set.image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        require_project_owner(self.db, current_user, image.project_id)

    def _validate_project_annotation_shape(self, project_id: int, annotations) -> None:
        from app.modules.projects.models import Project
        project = self.db.get(Project, project_id)
        if project and project.task_type == "POSE_ESTIMATION":
            if any(len(annotation.keypoints) != 17 for annotation in annotations):
                raise HTTPException(
                    status_code=400,
                    detail="Pose annotations require exactly 17 keypoints per person",
                )
