from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.annotations.models import AnnotationSet
from app.modules.datasets.models import DatasetVersion, DatasetVersionItem
from app.modules.datasets.schemas import DatasetVersionCreate
from app.modules.images.models import Image


class DatasetRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def next_version_number(self, project_id: int) -> int:
        current = self.db.scalar(
            select(func.max(DatasetVersion.version_number)).where(
                DatasetVersion.project_id == project_id
            )
        )
        return (current or 0) + 1

    def latest_approved_annotation_sets(self, project_id: int) -> list[AnnotationSet]:
        images = self.db.scalars(
            select(Image)
            .where(Image.project_id == project_id, Image.status != "ARCHIVED")
            .order_by(Image.image_id)
        )
        selected: list[AnnotationSet] = []
        for image in images:
            annotation_set = self.db.scalar(
                select(AnnotationSet)
                .where(
                    AnnotationSet.image_id == image.image_id,
                    AnnotationSet.status == "APPROVED",
                )
                .options(selectinload(AnnotationSet.annotations))
                .order_by(AnnotationSet.revision.desc())
                .limit(1)
            )
            if annotation_set is not None:
                selected.append(annotation_set)
        return selected

    def create_version(
        self,
        data: DatasetVersionCreate,
        version_number: int,
        annotation_sets: list[AnnotationSet],
        created_by: int,
    ) -> DatasetVersion:
        dataset_version = DatasetVersion(
            project_id=data.project_id,
            version_number=version_number,
            preprocess_config=data.preprocess_config,
            augmentation_config=data.augmentation_config,
            created_by=created_by,
        )
        self.db.add(dataset_version)
        self.db.flush()

        splits = self._item_splits(data, len(annotation_sets))
        self.db.add_all(
            [
                DatasetVersionItem(
                    version_id=dataset_version.version_id,
                    image_id=annotation_set.image_id,
                    annotation_set_id=annotation_set.annotation_set_id,
                    split=splits[item_index],
                )
                for item_index, annotation_set in enumerate(annotation_sets)
            ]
        )
        self.db.flush()
        return dataset_version

    def _item_splits(self, data: DatasetVersionCreate, item_count: int) -> list[str]:
        if not data.auto_split or item_count < 3:
            return [data.default_split.value] * item_count

        valid_count = max(1, round(item_count * 0.1))
        test_count = max(1, round(item_count * 0.1))
        if valid_count + test_count >= item_count:
            valid_count = 1
            test_count = 1
        train_count = item_count - valid_count - test_count
        return [
            *(["TRAIN"] * train_count),
            *(["VALID"] * valid_count),
            *(["TEST"] * test_count),
        ]

    def get(self, version_id: int) -> DatasetVersion | None:
        return self.db.scalar(
            select(DatasetVersion)
            .where(DatasetVersion.version_id == version_id)
            .options(selectinload(DatasetVersion.items))
        )

    def list_by_project(self, project_id: int) -> list[DatasetVersion]:
        return list(
            self.db.scalars(
                select(DatasetVersion)
                .where(DatasetVersion.project_id == project_id)
                .options(selectinload(DatasetVersion.items))
                .order_by(DatasetVersion.version_number)
            )
        )

    def delete(self, version_id: int) -> None:
        self.db.execute(delete(DatasetVersionItem).where(DatasetVersionItem.version_id == version_id))
        self.db.execute(delete(DatasetVersion).where(DatasetVersion.version_id == version_id))
