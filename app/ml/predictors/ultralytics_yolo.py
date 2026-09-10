from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.ml.predictors.base import Predictor


class UltralyticsYoloPredictor(Predictor):
    def predict(
        self,
        model_path: Path,
        image_paths: list[Path],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="ultralytics is not installed. Run: pip install -e .",
            ) from exc

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model weights not found: {model_path}")

        conf = float(config.get("confidence_threshold", config.get("conf", 0.25)))
        imgsz = config.get("imgsz")
        max_predictions = int(config.get("max_predictions", 20))
        model = YOLO(str(model_path))
        predict_kwargs = {"conf": conf, "verbose": False}
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz

        results = model.predict([str(path) for path in image_paths], **predict_kwargs)
        predictions: list[dict[str, Any]] = []
        for image_path, result in zip(image_paths, results, strict=False):
            height, width = result.orig_shape
            boxes = []
            keypoints_xy = getattr(getattr(result, "keypoints", None), "xyn", None)
            keypoints_conf = getattr(getattr(result, "keypoints", None), "conf", None)
            for index, box in enumerate(result.boxes):
                x_min, y_min, x_max, y_max = [float(value) for value in box.xyxy[0].tolist()]
                class_index = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                prediction = {
                    "class_index": class_index,
                    "x_min": self._clamp(x_min / width),
                    "y_min": self._clamp(y_min / height),
                    "x_max": self._clamp(x_max / width),
                    "y_max": self._clamp(y_max / height),
                    "confidence": confidence,
                }
                if keypoints_xy is not None and index < len(keypoints_xy):
                    xy_points = keypoints_xy[index].tolist()
                    confidence_points = keypoints_conf[index].tolist() if keypoints_conf is not None else []
                    prediction["keypoints"] = [
                        {
                            "x": self._clamp(float(point[0])),
                            "y": self._clamp(float(point[1])),
                            "visibility": 2 if point_conf >= 0.5 else 1 if point_conf >= 0.05 else 0,
                        }
                        for point, point_conf in zip(xy_points, confidence_points or [1.0] * len(xy_points), strict=True)
                    ]
                boxes.append(prediction)
            boxes.sort(key=lambda item: item["confidence"], reverse=True)
            boxes = boxes[:max_predictions]
            predictions.append({"image_path": image_path, "boxes": boxes})
        return predictions

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
