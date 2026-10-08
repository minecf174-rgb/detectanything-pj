from hashlib import sha256
from io import BytesIO
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from fastapi import HTTPException, UploadFile
from PIL import Image as PILImage
from PIL import ImageOps
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.core.config.settings import get_settings
from app.core.authz import require_project_owner
from app.core.storage.service import LocalStorageService
from app.modules.annotations.models import Annotation, AnnotationKeypoint, AnnotationSet
from app.modules.auto_label.models import AutoLabelJobItem
from app.modules.classes.models import LabelClass
from app.modules.datasets.models import DatasetVersionItem
from app.modules.images.models import Image
from app.modules.images.repository import ImageRepository
from app.modules.images.schemas import ImageBulkDeleteRequest, ImageBulkStatusRequest
from app.modules.projects.models import User


class ImageService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = ImageRepository(db)
        self.storage = LocalStorageService(get_settings().storage_root)

    async def upload_image(self, project_id: int, file: UploadFile, current_user: User):
        require_project_owner(self.db, current_user, project_id)

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        file_hash = sha256(data).hexdigest()
        width, height = self._read_image_size(data)
        suffix = Path(file.filename or "image").suffix.lower() or ".jpg"
        object_key = f"projects/project_{project_id}/images/{uuid4().hex}{suffix}"
        self.storage.save(object_key, data)

        image = self.repository.create(
            project_id=project_id,
            original_filename=file.filename,
            object_key=object_key,
            file_hash=file_hash,
            width=width,
            height=height,
            status="UPLOADED",
        )
        self.db.commit()
        self.db.refresh(image)
        return image

    def list_project_images(
        self, project_id: int, include_archived: bool, current_user: User
    ):
        require_project_owner(self.db, current_user, project_id)
        return self.repository.list_by_project(project_id, include_archived)

    def update_images_status(self, data: ImageBulkStatusRequest, current_user: User):
        require_project_owner(self.db, current_user, data.project_id)
        image_ids = list(dict.fromkeys(data.image_ids))
        images = list(
            self.db.query(Image)
            .filter(Image.image_id.in_(image_ids))
            .all()
        )
        found_ids = {image.image_id for image in images}
        missing_ids = [image_id for image_id in image_ids if image_id not in found_ids]
        if missing_ids:
            raise HTTPException(status_code=404, detail=f"Images not found: {missing_ids}")
        if any(image.project_id != data.project_id for image in images):
            raise HTTPException(status_code=400, detail="Images belong to another project")
        updated_count = 0
        for image in images:
            if image.status == data.status:
                continue
            image.status = data.status
            updated_count += 1
        self.db.commit()
        return {"updated_count": updated_count, "status": data.status}

    def delete_images_permanently(self, data: ImageBulkDeleteRequest, current_user: User):
        require_project_owner(self.db, current_user, data.project_id)
        image_ids = list(dict.fromkeys(data.image_ids))
        images = list(self.db.scalars(select(Image).where(Image.image_id.in_(image_ids))))
        found_ids = {image.image_id for image in images}
        missing_ids = [image_id for image_id in image_ids if image_id not in found_ids]
        if missing_ids:
            raise HTTPException(status_code=404, detail=f"Images not found: {missing_ids}")
        if any(image.project_id != data.project_id for image in images):
            raise HTTPException(status_code=400, detail="Images belong to another project")
        active_ids = [image.image_id for image in images if image.status != "ARCHIVED"]
        if active_ids:
            raise HTTPException(
                status_code=409,
                detail=f"Archive images before permanent deletion: {active_ids}",
            )
        object_keys = [image.object_key for image in images]
        preview_paths = [self._preview_path(image) for image in images]

        annotation_set_ids = list(
            self.db.scalars(
                select(AnnotationSet.annotation_set_id).where(AnnotationSet.image_id.in_(image_ids))
            )
        )
        self.db.execute(delete(DatasetVersionItem).where(DatasetVersionItem.image_id.in_(image_ids)))
        self.db.execute(delete(AutoLabelJobItem).where(AutoLabelJobItem.image_id.in_(image_ids)))
        if annotation_set_ids:
            annotation_ids = list(
                self.db.scalars(
                    select(Annotation.annotation_id).where(Annotation.annotation_set_id.in_(annotation_set_ids))
                )
            )
            if annotation_ids:
                self.db.execute(delete(AnnotationKeypoint).where(AnnotationKeypoint.annotation_id.in_(annotation_ids)))
            self.db.execute(delete(Annotation).where(Annotation.annotation_set_id.in_(annotation_set_ids)))
            self.db.execute(
                update(AnnotationSet)
                .where(AnnotationSet.parent_annotation_set_id.in_(annotation_set_ids))
                .values(parent_annotation_set_id=None)
            )
            self.db.execute(
                delete(AnnotationSet).where(AnnotationSet.annotation_set_id.in_(annotation_set_ids))
            )
        self.db.execute(delete(Image).where(Image.image_id.in_(image_ids)))
        self.db.commit()

        for object_key in object_keys:
            self.storage.delete(object_key)
        for preview_path in preview_paths:
            preview_path.unlink(missing_ok=True)
        return {"deleted_count": len(images)}

    async def import_yolo_zip(
        self,
        project_id: int,
        current_user: User,
        file: UploadFile,
        class_names: str | None = None,
        max_labeled_images: int | None = None,
        class_indexes: str | None = None,
        import_remaining_images: bool = True,
    ):
        project = require_project_owner(self.db, current_user, project_id)

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty zip file")

        try:
            with ZipFile(BytesIO(data)) as archive:
                names = [name for name in archive.namelist() if self._is_usable_zip_member(name)]
                inferred_class_names = self._infer_yolo_classes_from_labels(archive, names)
                provided_class_names = self._provided_class_names(class_names)
                parsed_class_names = provided_class_names.copy()
                selected_class_indexes = self._provided_class_indexes(class_indexes)
                if not parsed_class_names:
                    parsed_class_names = self._read_yolo_classes(archive, names, required=False)
                if not parsed_class_names:
                    parsed_class_names = inferred_class_names
                if len(parsed_class_names) < len(inferred_class_names):
                    parsed_class_names = [
                        *parsed_class_names,
                        *inferred_class_names[len(parsed_class_names):],
                    ]
                class_names_by_source_index = self._select_class_names(
                    parsed_class_names,
                    inferred_class_names,
                    selected_class_indexes,
                    provided_class_names,
                )
                classes = self._ensure_classes(project_id, list(class_names_by_source_index.values()))
                classes_by_source_index = {
                    source_index: classes[item_index]
                    for item_index, source_index in enumerate(class_names_by_source_index)
                }
                image_names = [
                    name for name in names
                    if self._is_image_name(name) and "/images/" in f"/{name}"
                ]
                if not image_names:
                    image_names = [name for name in names if self._is_image_name(name)]
                if max_labeled_images is not None and max_labeled_images < 0:
                    raise HTTPException(status_code=400, detail="max_labeled_images must be >= 0")

                prepared_images: list[tuple[str, list[dict]]] = []
                selected_labeled_images = 0
                for image_name in image_names:
                    label_name = self._matching_label_name(image_name, names)
                    boxes = [] if label_name is None else self._read_yolo_label_file(
                        archive.read(label_name).decode("utf-8"),
                        classes_by_source_index,
                        project.task_type,
                    )
                    include_labels = bool(boxes) and (
                        max_labeled_images is None or selected_labeled_images < max_labeled_images
                    )
                    if include_labels:
                        selected_labeled_images += 1
                    if include_labels or import_remaining_images:
                        prepared_images.append((image_name, boxes if include_labels else []))

                if not prepared_images:
                    raise HTTPException(status_code=400, detail="No labeled images matched the import settings")

                imported_images = 0
                imported_sets = 0
                imported_boxes = 0
                for image_name, boxes in prepared_images:
                    image_data = archive.read(image_name)
                    width, height = self._read_image_size(image_data)
                    suffix = Path(image_name).suffix.lower() or ".jpg"
                    object_key = f"projects/project_{project_id}/images/{uuid4().hex}{suffix}"
                    self.storage.save(object_key, image_data)
                    image = self.repository.create(
                        project_id=project_id,
                        original_filename=Path(image_name).name,
                        object_key=object_key,
                        file_hash=sha256(image_data).hexdigest(),
                        width=width,
                        height=height,
                        status="UPLOADED",
                    )
                    imported_images += 1

                    if not boxes:
                        continue

                    annotation_set = AnnotationSet(
                        image_id=image.image_id,
                        revision=1,
                        source_type="HUMAN",
                        status="APPROVED",
                        created_by=current_user.user_id,
                    )
                    self.db.add(annotation_set)
                    self.db.flush()
                    for box in boxes:
                        annotation = Annotation(
                            annotation_set_id=annotation_set.annotation_set_id,
                            class_id=box["class_id"],
                            x_min=box["x_min"],
                            y_min=box["y_min"],
                            x_max=box["x_max"],
                            y_max=box["y_max"],
                            confidence=None,
                        )
                        self.db.add(annotation)
                        self.db.flush()
                        self.db.add_all([
                            AnnotationKeypoint(
                                annotation_id=annotation.annotation_id,
                                keypoint_index=index,
                                x=keypoint["x"],
                                y=keypoint["y"],
                                visibility=keypoint["visibility"],
                            )
                            for index, keypoint in enumerate(box.get("keypoints", []))
                        ])
                    imported_sets += 1
                    imported_boxes += len(boxes)

                self.db.commit()
                return {
                    "imported_images": imported_images,
                    "imported_annotation_sets": imported_sets,
                    "imported_boxes": imported_boxes,
                    "classes": [label_class.name for label_class in classes],
                }
        except BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Invalid zip file") from exc
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Label files must be UTF-8 text") from exc

    async def inspect_archive(self, file: UploadFile):
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty zip file")
        try:
            with ZipFile(BytesIO(data)) as archive:
                names = [name for name in archive.namelist() if self._is_usable_zip_member(name)]
                image_names = [name for name in names if self._is_image_name(name)]
                if not image_names:
                    raise HTTPException(status_code=400, detail="Zip does not contain supported images")
                label_names = [
                    name for name in names
                    if Path(name).suffix.lower() == ".txt" and "/labels/" in f"/{name}"
                ]
                has_yolo_metadata = any(
                    Path(name).name in {"classes.txt", "data.yaml", "data.yml"} for name in names
                )
                is_yolo = bool(label_names or has_yolo_metadata)
                class_names = self._read_yolo_classes(archive, names, required=False) if is_yolo else []
                if is_yolo and not class_names and label_names:
                    class_names = self._infer_yolo_classes_from_labels(archive, names)
                labeled_image_count = 0
                if is_yolo:
                    for image_name in image_names:
                        label_name = self._matching_label_name(image_name, names)
                        if label_name is None:
                            continue
                        label_text = archive.read(label_name).decode("utf-8")
                        if self._label_file_has_annotations(label_text):
                            labeled_image_count += 1
                return {
                    "archive_type": "YOLO" if is_yolo else "IMAGES",
                    "image_count": len(image_names),
                    "labeled_image_count": labeled_image_count,
                    "unlabeled_image_count": len(image_names) - labeled_image_count,
                    "class_names": class_names,
                }
        except BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Invalid zip file") from exc
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="YOLO metadata must be UTF-8 text") from exc

    async def import_image_archive(self, project_id: int, current_user: User, file: UploadFile):
        require_project_owner(self.db, current_user, project_id)
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty zip file")
        saved_keys: list[str] = []
        try:
            with ZipFile(BytesIO(data)) as archive:
                names = [name for name in archive.namelist() if self._is_usable_zip_member(name)]
                image_names = [name for name in names if self._is_image_name(name)]
                if not image_names:
                    raise HTTPException(status_code=400, detail="Zip does not contain supported images")
                for image_name in image_names:
                    image_data = archive.read(image_name)
                    width, height = self._read_image_size(image_data)
                    suffix = Path(image_name).suffix.lower() or ".jpg"
                    object_key = f"projects/project_{project_id}/images/{uuid4().hex}{suffix}"
                    self.storage.save(object_key, image_data)
                    saved_keys.append(object_key)
                    self.repository.create(
                        project_id=project_id,
                        original_filename=Path(image_name).name,
                        object_key=object_key,
                        file_hash=sha256(image_data).hexdigest(),
                        width=width,
                        height=height,
                        status="UPLOADED",
                    )
                self.db.commit()
                return {
                    "imported_images": len(image_names),
                    "imported_annotation_sets": 0,
                    "imported_boxes": 0,
                    "classes": [],
                }
        except BadZipFile as exc:
            self.db.rollback()
            raise HTTPException(status_code=400, detail="Invalid zip file") from exc
        except Exception:
            self.db.rollback()
            for object_key in saved_keys:
                self.storage.delete(object_key)
            raise

    def get_image_file_path(self, image_id: int, current_user: User) -> Path:
        _, path = self._authorized_image_path(image_id, current_user)
        return path

    def get_image_preview_file_path(self, image_id: int, current_user: User) -> Path:
        image, source_path = self._authorized_image_path(image_id, current_user)
        preview_path = self._preview_path(image)
        if preview_path.exists():
            return preview_path
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(dir=preview_path.parent, suffix=".webp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            with PILImage.open(source_path) as source:
                preview = ImageOps.exif_transpose(source)
                preview.thumbnail((768, 768), PILImage.Resampling.LANCZOS)
                if preview.mode not in {"RGB", "RGBA"}:
                    preview = preview.convert("RGBA" if "A" in preview.getbands() or "transparency" in preview.info else "RGB")
                preview.save(temporary_path, format="WEBP", quality=78, method=4)
            replace(temporary_path, preview_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return preview_path

    def _preview_path(self, image: Image) -> Path:
        version = sha256(f"{image.object_key}:{image.file_hash}:webp-768-v1".encode()).hexdigest()[:20]
        return self.storage.root / "previews" / f"project_{image.project_id}" / f"image_{image.image_id}_{version}.webp"

    def _authorized_image_path(self, image_id: int, current_user: User) -> tuple[Image, Path]:
        image = self.repository.get(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        require_project_owner(self.db, current_user, image.project_id)
        path = self.storage.root / image.object_key
        if not path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        return image, path

    def _read_image_size(self, data: bytes) -> tuple[int | None, int | None]:
        try:
            with PILImage.open(BytesIO(data)) as image:
                return image.width, image.height
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid image file") from exc

    def _is_usable_zip_member(self, name: str) -> bool:
        path = Path(name)
        return not (
            name.endswith("/")
            or name.startswith("__MACOSX/")
            or "/__MACOSX/" in name
            or path.name.startswith("._")
            or path.name == ".DS_Store"
        )

    def _provided_class_names(self, class_names: str | None) -> list[str]:
        if not class_names:
            return []
        return [name.strip() for name in class_names.split(",") if name.strip()]

    def _provided_class_indexes(self, class_indexes: str | None) -> list[int]:
        if not class_indexes:
            return []
        indexes: list[int] = []
        for raw_index in class_indexes.split(","):
            raw_index = raw_index.strip()
            if not raw_index:
                continue
            try:
                index = int(raw_index)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="class_indexes must be comma-separated integers") from exc
            if index < 0:
                raise HTTPException(status_code=400, detail="class_indexes must be >= 0")
            if index not in indexes:
                indexes.append(index)
        return indexes

    def _read_yolo_classes(self, archive: ZipFile, names: list[str], required: bool = True) -> list[str]:
        classes_file = next((name for name in names if Path(name).name == "classes.txt"), None)
        if classes_file:
            classes = [
                line.strip()
                for line in archive.read(classes_file).decode("utf-8").splitlines()
                if line.strip()
            ]
            if classes:
                return classes

        data_yaml = next((name for name in names if Path(name).name in {"data.yaml", "data.yml"}), None)
        if data_yaml:
            text = archive.read(data_yaml).decode("utf-8")
            parsed = self._parse_yolo_names_from_yaml_text(text)
            if parsed:
                return parsed

        if required:
            raise HTTPException(status_code=400, detail="Zip must include classes.txt or data.yaml names")
        return []

    def _infer_yolo_classes_from_labels(self, archive: ZipFile, names: list[str]) -> list[str]:
        max_class_index = -1
        for name in names:
            if Path(name).suffix.lower() != ".txt" or "/labels/" not in f"/{name}":
                continue
            text = archive.read(name).decode("utf-8")
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    max_class_index = max(max_class_index, int(float(parts[0])))
        if max_class_index < 0:
            raise HTTPException(status_code=400, detail="No YOLO label boxes found")
        return [f"class_{index}" for index in range(max_class_index + 1)]

    def _select_class_names(
        self,
        parsed_class_names: list[str],
        inferred_class_names: list[str],
        selected_class_indexes: list[int],
        provided_class_names: list[str],
    ) -> dict[int, str]:
        if not selected_class_indexes:
            selected_class_indexes = list(range(len(inferred_class_names)))

        max_available_index = len(inferred_class_names) - 1
        invalid_indexes = [index for index in selected_class_indexes if index > max_available_index]
        if invalid_indexes:
            raise HTTPException(
                status_code=400,
                detail=f"class_indexes not found in labels: {', '.join(str(index) for index in invalid_indexes)}",
            )

        if provided_class_names and len(provided_class_names) == len(selected_class_indexes):
            return {
                source_index: provided_class_names[item_index]
                for item_index, source_index in enumerate(selected_class_indexes)
            }

        return {
            source_index: (
                parsed_class_names[source_index]
                if source_index < len(parsed_class_names)
                else inferred_class_names[source_index]
            )
            for source_index in selected_class_indexes
        }

    def _parse_yolo_names_from_yaml_text(self, text: str) -> list[str]:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("names:"):
                continue
            value = stripped.removeprefix("names:").strip()
            if value.startswith("[") and value.endswith("]"):
                return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            if value.startswith("{") and value.endswith("}"):
                entries = []
                for item in value[1:-1].split(","):
                    if ":" not in item:
                        continue
                    entries.append(item.split(":", 1)[1].strip().strip("'\""))
                return [entry for entry in entries if entry]
            names = []
            for next_line in lines[index + 1:]:
                if not next_line.startswith(" ") and not next_line.startswith("-"):
                    break
                clean = next_line.strip()
                if clean.startswith("-"):
                    names.append(clean[1:].strip().strip("'\""))
            return [name for name in names if name]
        return []

    def _ensure_classes(self, project_id: int, class_names: list[str]) -> list[LabelClass]:
        existing = {
            label_class.name: label_class
            for label_class in self.db.query(LabelClass).filter(LabelClass.project_id == project_id).all()
        }
        classes: list[LabelClass] = []
        next_index = max((label_class.class_index for label_class in existing.values()), default=-1) + 1
        palette = ["#16a34a", "#2563eb", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]
        for class_name in class_names:
            if class_name in existing:
                classes.append(existing[class_name])
                continue
            label_class = LabelClass(
                project_id=project_id,
                name=class_name,
                class_index=next_index,
                color=palette[next_index % len(palette)],
            )
            self.db.add(label_class)
            self.db.flush()
            classes.append(label_class)
            next_index += 1
        return classes

    def _is_image_name(self, name: str) -> bool:
        return Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _matching_label_name(self, image_name: str, names: list[str]) -> str | None:
        image_path = Path(image_name)
        candidates = [
            str(image_path.with_suffix(".txt")),
            str(Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")),
            f"labels/{image_path.stem}.txt",
        ]
        names_set = set(names)
        for candidate in candidates:
            if candidate in names_set:
                return candidate
        return next((name for name in names if Path(name).stem == image_path.stem and Path(name).suffix == ".txt"), None)

    def _read_yolo_label_file(
        self,
        text: str,
        classes_by_source_index: dict[int, LabelClass],
        task_type: str,
    ) -> list[dict]:
        boxes = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_index = int(float(parts[0]))
            label_class = classes_by_source_index.get(class_index)
            if label_class is None:
                continue

            values = [float(value) for value in parts[1:]]
            keypoints: list[dict] = []
            if task_type == "POSE_ESTIMATION":
                if len(values) != 4 + (17 * 3):
                    continue
                x_center, y_center, width, height = values[:4]
                x_min = self._clamp(x_center - width / 2)
                y_min = self._clamp(y_center - height / 2)
                x_max = self._clamp(x_center + width / 2)
                y_max = self._clamp(y_center + height / 2)
                pose_values = values[4:]
                for index in range(17):
                    x, y, raw_visibility = pose_values[index * 3:(index + 1) * 3]
                    visibility = int(raw_visibility)
                    if raw_visibility != visibility or visibility not in {0, 1, 2}:
                        keypoints = []
                        break
                    keypoints.append({
                        "x": self._clamp(x),
                        "y": self._clamp(y),
                        "visibility": visibility,
                    })
                if len(keypoints) != 17:
                    continue
            elif len(values) == 4:
                x_center, y_center, width, height = values
                x_min = self._clamp(x_center - width / 2)
                y_min = self._clamp(y_center - height / 2)
                x_max = self._clamp(x_center + width / 2)
                y_max = self._clamp(y_center + height / 2)
            elif len(values) >= 6 and len(values) % 2 == 0:
                # YOLO segmentation stores polygon points after the class index.
                # This application is box-based, so retain the polygon's bounding box.
                x_values = values[0::2]
                y_values = values[1::2]
                x_min = self._clamp(min(x_values))
                y_min = self._clamp(min(y_values))
                x_max = self._clamp(max(x_values))
                y_max = self._clamp(max(y_values))
            else:
                continue

            if x_min >= x_max or y_min >= y_max:
                continue
            boxes.append(
                {
                    "class_id": label_class.class_id,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "keypoints": keypoints,
                }
            )
        return boxes

    def _label_file_has_annotations(self, text: str) -> bool:
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                [float(value) for value in parts[:5]]
            except ValueError:
                continue
            return True
        return False

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
