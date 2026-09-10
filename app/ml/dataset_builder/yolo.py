from pathlib import Path
from shutil import copy2, rmtree

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config.settings import get_settings
from app.ml.dataset_builder.builder import DatasetBuilder
from app.modules.annotations.models import Annotation, AnnotationSet
from app.modules.classes.models import LabelClass
from app.modules.datasets.models import DatasetVersion, DatasetVersionItem


class YoloDatasetBuilder(DatasetBuilder):
    def __init__(self, db: Session) -> None:
        self.db = db
        settings = get_settings()
        self.storage_root = Path(settings.storage_root)
        self.temp_root = Path(settings.temp_root)

    def build(self, dataset_version_id: int, output_dir: Path | None = None) -> Path:
        dataset_version = self._get_dataset_version(dataset_version_id)
        output_dir = output_dir or self.temp_root / "training" / f"dataset_version_{dataset_version_id}"

        if output_dir.exists():
            rmtree(output_dir)
        self._create_yolo_dirs(output_dir)

        classes = self._get_classes(dataset_version.project_id)
        class_index_by_id = {label_class.class_id: label_class.class_index for label_class in classes}

        image_count = 0
        label_count = 0
        split_counts = {"train": 0, "val": 0, "test": 0}
        for item in dataset_version.items:
            split_dir = self._split_dir(item.split)
            image_target = self._copy_image(item, output_dir, split_dir)
            labels = self._build_label_lines(
                item.annotation_set, class_index_by_id, dataset_version.project.task_type
            )
            label_target = output_dir / "labels" / split_dir / f"{image_target.stem}.txt"
            label_target.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            image_count += 1
            label_count += len(labels)
            split_counts[split_dir] += 1

        self._write_data_yaml(output_dir, classes, split_counts, dataset_version.project.task_type)
        (output_dir / "BUILD_SUMMARY.txt").write_text(
            f"dataset_version_id={dataset_version_id}\n"
            f"image_count={image_count}\n"
            f"label_count={label_count}\n",
            encoding="utf-8",
        )
        return output_dir

    def _get_dataset_version(self, dataset_version_id: int) -> DatasetVersion:
        dataset_version = self.db.scalar(
            select(DatasetVersion)
            .where(DatasetVersion.version_id == dataset_version_id)
            .options(
                selectinload(DatasetVersion.items)
                .selectinload(DatasetVersionItem.image),
                selectinload(DatasetVersion.items)
                .selectinload(DatasetVersionItem.annotation_set)
                .selectinload(AnnotationSet.annotations)
                .selectinload(Annotation.keypoints),
            )
        )
        if dataset_version is None:
            raise HTTPException(status_code=404, detail="Dataset version not found")
        if not dataset_version.items:
            raise HTTPException(status_code=400, detail="Dataset version has no items")
        return dataset_version

    def _get_classes(self, project_id: int) -> list[LabelClass]:
        classes = list(
            self.db.scalars(
                select(LabelClass)
                .where(LabelClass.project_id == project_id)
                .order_by(LabelClass.class_index)
            )
        )
        if not classes:
            raise HTTPException(status_code=400, detail="Project has no classes")
        return classes

    def _create_yolo_dirs(self, output_dir: Path) -> None:
        for split in ("train", "val", "test"):
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _copy_image(self, item: DatasetVersionItem, output_dir: Path, split_dir: str) -> Path:
        source = self.storage_root / item.image.object_key
        if not source.exists():
            raise HTTPException(status_code=500, detail=f"Image file missing: {item.image.object_key}")

        suffix = source.suffix or Path(item.image.original_filename or "").suffix or ".jpg"
        target = output_dir / "images" / split_dir / f"image_{item.image_id}{suffix}"
        copy2(source, target)
        return target

    def _build_label_lines(
        self,
        annotation_set: AnnotationSet,
        class_index_by_id: dict[int, int],
        task_type: str,
    ) -> list[str]:
        lines: list[str] = []
        for annotation in annotation_set.annotations:
            class_index = class_index_by_id.get(annotation.class_id)
            if class_index is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Class {annotation.class_id} is not part of this project",
                )
            x_center = (annotation.x_min + annotation.x_max) / 2
            y_center = (annotation.y_min + annotation.y_max) / 2
            width = annotation.x_max - annotation.x_min
            height = annotation.y_max - annotation.y_min
            line = (
                f"{class_index} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )
            if task_type == "POSE_ESTIMATION":
                keypoints = sorted(annotation.keypoints, key=lambda item: item.keypoint_index)
                if len(keypoints) != 17 or [item.keypoint_index for item in keypoints] != list(range(17)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Pose annotation {annotation.annotation_id} must include 17 ordered keypoints",
                    )
                line += " " + " ".join(
                    f"{item.x:.6f} {item.y:.6f} {item.visibility}" for item in keypoints
                )
            lines.append(line)
        return lines

    def _write_data_yaml(
        self,
        output_dir: Path,
        classes: list[LabelClass],
        split_counts: dict[str, int],
        task_type: str,
    ) -> None:
        names = ", ".join(f'"{label_class.name}"' for label_class in classes)
        val_path = "images/val" if split_counts["val"] else "images/train"
        test_path = "images/test" if split_counts["test"] else "images/train"
        data_yaml = (
            f"path: {output_dir.resolve()}\n"
            "train: images/train\n"
            f"val: {val_path}\n"
            f"test: {test_path}\n"
            f"nc: {len(classes)}\n"
            f"names: [{names}]\n"
        )
        if task_type == "POSE_ESTIMATION":
            data_yaml += "kpt_shape: [17, 3]\n"
        (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    def _split_dir(self, split: str) -> str:
        if split == "VALID":
            return "val"
        return split.lower()
