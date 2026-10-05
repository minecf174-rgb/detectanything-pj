from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import is_zipfile

from fastapi import HTTPException
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.authz import require_project_owner
from app.core.config.settings import get_settings
from app.core.storage.service import LocalStorageService
from app.ml.predictors.ultralytics_yolo import UltralyticsYoloPredictor
from app.modules.auto_label.models import AutoLabelJob
from app.modules.classes.models import LabelClass
from app.modules.models.class_mapping import model_class_mapping
from app.modules.models.models import Model, ModelArtifact
from app.modules.models.repository import ModelRepository
from app.modules.projects.models import User
from app.modules.training.models import TrainingRun


class ModelService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = ModelRepository(db)
        self.storage = LocalStorageService(get_settings().storage_root)

    def import_model(self, project_id: int, data: bytes, filename: str | None, current_user: User):
        project = require_project_owner(self.db, current_user, project_id)
        if Path(filename or "").suffix.lower() != ".pt":
            raise HTTPException(status_code=400, detail="Choose an Ultralytics YOLO .pt weights file")
        if not data:
            raise HTTPException(status_code=400, detail="Model file is empty")
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Model file must be 200 MB or smaller")
        if not is_zipfile(BytesIO(data)):
            raise HTTPException(status_code=400, detail="Model file is not a supported PyTorch checkpoint")

        temp_dir = Path(get_settings().temp_root) / "model-imports"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{uuid4().hex}.pt"
        try:
            temp_path.write_bytes(data)
            task, names, safe_data = self._inspect_uploaded_weights(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        expected_task = "pose" if project.task_type == "POSE_ESTIMATION" else "detect"
        if task != expected_task:
            raise HTTPException(
                status_code=400,
                detail=f"This is a {task} model; project requires a {expected_task} model",
            )

        existing_classes = list(self.db.scalars(select(LabelClass).where(LabelClass.project_id == project_id)))
        by_index = {item.class_index: item for item in existing_classes}
        by_name = {item.name: item for item in existing_classes}
        for index, name in enumerate(names):
            indexed = by_index.get(index)
            named = by_name.get(name)
            if indexed is not None and indexed.name != name:
                raise HTTPException(status_code=409, detail=(
                    f"Class {index} is '{indexed.name}' in this project but '{name}' in the model. "
                    "Use a project with matching classes; existing labels will not be changed."
                ))
            if named is not None and named.class_index != index:
                raise HTTPException(status_code=409, detail=(
                    f"Class '{name}' is index {named.class_index} in this project but {index} in the model. "
                    "Use a project with matching classes; existing labels will not be changed."
                ))

        model_type = "YOLO26_POSE" if task == "pose" else "YOLO26_DETECT"
        version = self.repository.next_version(project_id)
        import_id = uuid4().hex
        object_key = f"models/project_{project_id}/import_{import_id}/weights.pt"
        created_classes: list[str] = []
        committed = False
        try:
            colors = ("#e11d48", "#f97316", "#f59e0b", "#16a34a", "#0d9488", "#0891b2", "#2563eb", "#7c3aed")
            for index, name in enumerate(names):
                if index not in by_index:
                    self.db.add(LabelClass(project_id=project_id, name=name, class_index=index, color=colors[index % len(colors)]))
                    created_classes.append(name)
            model = Model(
                project_id=project_id,
                training_run_id=None,
                uploaded_by_user_id=current_user.user_id,
                source_type="IMPORTED",
                model_type=model_type,
                class_names=names,
                version=version,
                weights_object_key=object_key,
            )
            self.db.add(model)
            self.db.flush()
            self.db.add(ModelArtifact(model_id=model.model_id, artifact_type="BEST_PT", object_key=object_key))
            self.db.flush()
            self.storage.save(object_key, safe_data)
            self.db.commit()
            committed = True
            self.db.refresh(model)
            return {"model": model, "created_classes": created_classes}
        except IntegrityError as exc:
            self.db.rollback()
            raise HTTPException(status_code=409, detail="Model version or class changed during import; please retry") from exc
        except Exception:
            self.db.rollback()
            raise
        finally:
            if not committed:
                self.storage.delete_prefix(f"models/project_{project_id}/import_{import_id}")

    @staticmethod
    def _inspect_uploaded_weights(path: Path) -> tuple[str, list[str], bytes]:
        try:
            import torch
            from ultralytics.nn.tasks import _SafeLoad, guess_model_task, torch_safe_load

            if not _SafeLoad.SUPPORTED:
                raise ValueError("Safe PyTorch checkpoint loading is unavailable; update PyTorch and Ultralytics")
            checkpoint, _ = torch_safe_load(str(path), safe_only=True)
            loaded = checkpoint.get("ema") or checkpoint.get("model")
            if not isinstance(loaded, torch.nn.Module):
                raise ValueError("File does not contain an Ultralytics model")
            task = str(guess_model_task(loaded))
            raw_names = getattr(loaded, "names", None)
            if isinstance(raw_names, dict):
                indexed_names = {int(index): str(name).strip() for index, name in raw_names.items()}
                if sorted(indexed_names) != list(range(len(indexed_names))):
                    raise ValueError("Model class indexes must start at 0 without gaps")
                names = [indexed_names[index] for index in range(len(indexed_names))]
            elif isinstance(raw_names, (list, tuple)):
                names = [str(name).strip() for name in raw_names]
            else:
                raise ValueError("Model has no readable class names")
            if task not in {"detect", "pose"}:
                raise ValueError("Only object detection and pose models are supported")
            if not names or len(names) > 256 or any(not name or len(name) > 120 for name in names) or len(set(names)) != len(names):
                raise ValueError("Model class names must be non-empty, unique, and at most 120 characters each")
            if task == "pose":
                shape = getattr(loaded, "yaml", {}).get("kpt_shape")
                if shape and int(shape[0]) != 17:
                    raise ValueError("Pose projects require 17 keypoints per person")
            safe_buffer = BytesIO()
            torch.save({"model": loaded, "train_args": {"task": task}}, safe_buffer)
            return task, names, safe_buffer.getvalue()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not inspect YOLO model: {exc}") from exc

    def get_model(self, model_id: int, current_user: User):
        model = self.repository.get(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        require_project_owner(self.db, current_user, model.project_id)
        return self._with_weights_exists(model)

    def list_project_models(self, project_id: int, current_user: User):
        require_project_owner(self.db, current_user, project_id)
        return self.repository.list_by_project(project_id)

    def get_weights_file_path(self, model_id: int, current_user: User):
        model = self.repository.get(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        require_project_owner(self.db, current_user, model.project_id)
        path = self.storage.root / model.weights_object_key
        if not path.exists():
            raise HTTPException(status_code=404, detail="Model weights file not found")
        return path

    def get_download_file_path(self, model_id: int, export_format: str, current_user: User) -> Path:
        weights_path = self.get_weights_file_path(model_id, current_user)
        if export_format == "pt":
            return weights_path
        try:
            from ultralytics import YOLO

            exported = Path(YOLO(str(weights_path)).export(format=export_format, device="cpu"))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not export model as {export_format}: {exc}",
            ) from exc
        if not exported.exists():
            raise HTTPException(status_code=500, detail="Model export did not create a file")
        return exported

    def delete_model(self, model_id: int, current_user: User) -> None:
        model = self.repository.get(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        require_project_owner(self.db, current_user, model.project_id)

        active_auto_label_job = self.db.scalar(
            select(AutoLabelJob.job_id).where(
                AutoLabelJob.model_id == model_id,
                AutoLabelJob.status.in_({"QUEUED", "RUNNING"}),
            ).limit(1)
        )
        if active_auto_label_job is not None:
            raise HTTPException(
                status_code=409,
                detail="Model is being used by a queued or running auto-label job",
            )

        active_training_run = self.db.scalar(
            select(TrainingRun.training_run_id).where(
                TrainingRun.project_id == model.project_id,
                TrainingRun.base_model == f"model:{model_id}",
                TrainingRun.status.in_({"QUEUED", "RUNNING"}),
            ).limit(1)
        )
        if active_training_run is not None:
            raise HTTPException(
                status_code=409,
                detail="Model is being used by a queued or running training job",
            )

        model_directory = Path(model.weights_object_key).parent.as_posix()
        self.storage.delete_prefix(model_directory)
        model.deleted_at = datetime.now(UTC)
        self.db.commit()

    def predict_uploaded_image(
        self,
        model_id: int,
        image_data: bytes,
        filename: str | None,
        confidence_threshold: float,
        current_user: User,
    ):
        result = self.predict_uploaded_images(
            model_id=model_id,
            uploads=[(image_data, filename)],
            confidence_threshold=confidence_threshold,
            current_user=current_user,
        )
        return result["results"][0]

    def predict_uploaded_images(
        self,
        model_id: int,
        uploads: list[tuple[bytes, str | None]],
        confidence_threshold: float,
        current_user: User,
    ):
        if not uploads:
            raise HTTPException(status_code=400, detail="Choose at least one test image")
        if len(uploads) > 100:
            raise HTTPException(status_code=400, detail="Test at most 100 images at once")
        if any(not image_data for image_data, _ in uploads):
            raise HTTPException(status_code=400, detail="Empty image file")
        if any(len(image_data) > 25 * 1024 * 1024 for image_data, _ in uploads):
            raise HTTPException(status_code=413, detail="Each test image must be 25 MB or smaller")

        model = self.repository.get(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        require_project_owner(self.db, current_user, model.project_id)
        model_path = self.storage.root / model.weights_object_key
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model weights file not found")

        temp_dir = Path(get_settings().temp_root) / "model-tests"
        temp_dir.mkdir(parents=True, exist_ok=True)
        validated_uploads: list[tuple[str | None, int, int, Path]] = []
        try:
            for image_data, filename in uploads:
                image_width, image_height = self._read_image_size(image_data)
                suffix = Path(filename or "test-image.jpg").suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    suffix = ".jpg"
                temp_path = temp_dir / f"{uuid4().hex}{suffix}"
                temp_path.write_bytes(image_data)
                validated_uploads.append((filename, image_width, image_height, temp_path))
            predictions = UltralyticsYoloPredictor().predict(
                model_path,
                [upload[3] for upload in validated_uploads],
                {"confidence_threshold": confidence_threshold, "max_predictions": 100},
            )
        finally:
            for _, _, _, temp_path in validated_uploads:
                temp_path.unlink(missing_ok=True)

        classes = list(
            self.db.scalars(
                select(LabelClass)
                .where(LabelClass.project_id == model.project_id)
                .order_by(LabelClass.class_index)
            )
        )
        classes_by_index = model_class_mapping(model, classes)
        results = []
        for (_filename, image_width, image_height, _), prediction in zip(validated_uploads, predictions, strict=True):
            boxes = []
            for box in prediction["boxes"]:
                label_class = classes_by_index.get(box["class_index"])
                if label_class is None:
                    continue
                boxes.append(
                    {
                        "class_id": label_class.class_id,
                        "class_index": label_class.class_index,
                        "class_name": label_class.name,
                        "x_min": box["x_min"],
                        "y_min": box["y_min"],
                        "x_max": box["x_max"],
                        "y_max": box["y_max"],
                        "confidence": box["confidence"],
                    }
                )
            results.append(
                {
                    "model_id": model.model_id,
                    "project_id": model.project_id,
                    "image_width": image_width,
                    "image_height": image_height,
                    "boxes": boxes,
                }
            )
        return {
            "model_id": model.model_id,
            "project_id": model.project_id,
            "results": results,
        }

    def stream_uploaded_images(
        self,
        model_id: int,
        uploads: list[tuple[bytes, str | None]],
        confidence_threshold: float,
        current_user: User,
    ):
        model = self.repository.get(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        require_project_owner(self.db, current_user, model.project_id)
        model_path = self.storage.root / model.weights_object_key
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model weights file not found")
        classes = list(
            self.db.scalars(
                select(LabelClass)
                .where(LabelClass.project_id == model.project_id)
                .order_by(LabelClass.class_index)
            )
        )
        return self._prediction_event_stream(
            model_path=model_path,
            model_id=model.model_id,
            project_id=model.project_id,
            uploads=uploads,
            confidence_threshold=confidence_threshold,
            classes_by_index=model_class_mapping(model, classes),
        )

    def predict_pretrained_pose_images(
        self,
        project_id: int,
        uploads: list[tuple[bytes, str | None]],
        confidence_threshold: float,
        current_user: User,
    ):
        if not uploads:
            raise HTTPException(status_code=400, detail="Choose at least one test image")
        if len(uploads) > 100:
            raise HTTPException(status_code=400, detail="Test at most 100 images at once")
        if any(not image_data for image_data, _ in uploads):
            raise HTTPException(status_code=400, detail="Empty image file")
        if any(len(image_data) > 25 * 1024 * 1024 for image_data, _ in uploads):
            raise HTTPException(status_code=413, detail="Each test image must be 25 MB or smaller")

        require_project_owner(self.db, current_user, project_id)
        pose_weights = Path(__file__).resolve().parents[3] / "yolo26n-pose.pt"
        if not pose_weights.exists():
            raise HTTPException(status_code=404, detail="YOLO26 pose pretrained weights are not installed")

        temp_dir = Path(get_settings().temp_root) / "model-tests"
        temp_dir.mkdir(parents=True, exist_ok=True)
        validated_uploads: list[tuple[str | None, int, int, Path]] = []
        try:
            for image_data, filename in uploads:
                image_width, image_height = self._read_image_size(image_data)
                suffix = Path(filename or "test-image.jpg").suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    suffix = ".jpg"
                temp_path = temp_dir / f"{uuid4().hex}{suffix}"
                temp_path.write_bytes(image_data)
                validated_uploads.append((filename, image_width, image_height, temp_path))
            predictions = UltralyticsYoloPredictor().predict(
                pose_weights,
                [upload[3] for upload in validated_uploads],
                {"confidence_threshold": confidence_threshold, "max_predictions": 20},
            )
        finally:
            for _, _, _, temp_path in validated_uploads:
                temp_path.unlink(missing_ok=True)

        results = []
        for (_filename, image_width, image_height, _), prediction in zip(validated_uploads, predictions, strict=True):
            results.append(
                {
                    "model_id": 0,
                    "project_id": project_id,
                    "image_width": image_width,
                    "image_height": image_height,
                    "boxes": [
                        {
                            "class_id": 0,
                            "class_index": box["class_index"],
                            "class_name": "person",
                            "x_min": box["x_min"],
                            "y_min": box["y_min"],
                            "x_max": box["x_max"],
                            "y_max": box["y_max"],
                            "confidence": box["confidence"],
                            "keypoints": box.get("keypoints", []),
                        }
                        for box in prediction["boxes"]
                    ],
                }
            )
        return {"model_id": 0, "project_id": project_id, "results": results}

    def stream_pretrained_pose_images(
        self,
        project_id: int,
        uploads: list[tuple[bytes, str | None]],
        confidence_threshold: float,
        current_user: User,
    ):
        require_project_owner(self.db, current_user, project_id)
        pose_weights = Path(__file__).resolve().parents[3] / "yolo26n-pose.pt"
        if not pose_weights.exists():
            raise HTTPException(status_code=404, detail="YOLO26 pose pretrained weights are not installed")
        return self._prediction_event_stream(
            model_path=pose_weights,
            model_id=0,
            project_id=project_id,
            uploads=uploads,
            confidence_threshold=confidence_threshold,
            classes_by_index=None,
        )

    def _prediction_event_stream(
        self,
        *,
        model_path: Path,
        model_id: int,
        project_id: int,
        uploads: list[tuple[bytes, str | None]],
        confidence_threshold: float,
        classes_by_index: dict[int, LabelClass] | None,
    ):
        if not uploads:
            raise HTTPException(status_code=400, detail="Choose at least one test image")
        if len(uploads) > 100:
            raise HTTPException(status_code=400, detail="Test at most 100 images at once")
        if any(not image_data for image_data, _ in uploads):
            raise HTTPException(status_code=400, detail="Empty image file")
        if any(len(image_data) > 25 * 1024 * 1024 for image_data, _ in uploads):
            raise HTTPException(status_code=413, detail="Each test image must be 25 MB or smaller")

        temp_dir = Path(get_settings().temp_root) / "model-tests"
        temp_dir.mkdir(parents=True, exist_ok=True)
        validated_uploads: list[tuple[str | None, int, int, Path]] = []
        try:
            for image_data, filename in uploads:
                image_width, image_height = self._read_image_size(image_data)
                suffix = Path(filename or "test-image.jpg").suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    suffix = ".jpg"
                temp_path = temp_dir / f"{uuid4().hex}{suffix}"
                temp_path.write_bytes(image_data)
                validated_uploads.append((filename, image_width, image_height, temp_path))
        except Exception:
            for _, _, _, temp_path in validated_uploads:
                temp_path.unlink(missing_ok=True)
            raise

        def events():
            try:
                predictions = UltralyticsYoloPredictor().predict_stream(
                    model_path,
                    [upload[3] for upload in validated_uploads],
                    {"confidence_threshold": confidence_threshold, "max_predictions": 100},
                )
                for index, (upload, prediction) in enumerate(
                    zip(validated_uploads, predictions, strict=True)
                ):
                    _, image_width, image_height, _ = upload
                    boxes = []
                    for box in prediction["boxes"]:
                        if classes_by_index is None:
                            boxes.append(
                                {
                                    "class_id": 0,
                                    "class_index": box["class_index"],
                                    "class_name": "person",
                                    **{key: box[key] for key in (
                                        "x_min", "y_min", "x_max", "y_max", "confidence"
                                    )},
                                    "keypoints": box.get("keypoints", []),
                                }
                            )
                            continue
                        label_class = classes_by_index.get(box["class_index"])
                        if label_class is None:
                            continue
                        boxes.append(
                            {
                                "class_id": label_class.class_id,
                                "class_index": label_class.class_index,
                                "class_name": label_class.name,
                                **{key: box[key] for key in (
                                    "x_min", "y_min", "x_max", "y_max", "confidence"
                                )},
                                "keypoints": box.get("keypoints", []),
                            }
                        )
                    yield {
                        "index": index,
                        "result": {
                            "model_id": model_id,
                            "project_id": project_id,
                            "image_width": image_width,
                            "image_height": image_height,
                            "boxes": boxes,
                        },
                    }
            finally:
                for _, _, _, temp_path in validated_uploads:
                    temp_path.unlink(missing_ok=True)

        return events()

    def _with_weights_exists(self, model):
        model.weights_exists = self.storage.exists(model.weights_object_key)
        return model

    def _read_image_size(self, image_data: bytes) -> tuple[int, int]:
        try:
            with PILImage.open(BytesIO(image_data)) as image:
                image.verify()
            with PILImage.open(BytesIO(image_data)) as image:
                return image.width, image.height
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid image file") from exc
