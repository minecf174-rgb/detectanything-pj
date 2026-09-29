from pathlib import Path
from shutil import copy2
from typing import Any

from fastapi import HTTPException

from app.ml.trainers.base import Trainer


class UltralyticsYoloTrainer(Trainer):
    def train(self, dataset_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="ultralytics is not installed. Run: pip install -e .",
            ) from exc

        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            raise HTTPException(status_code=400, detail=f"Missing data.yaml: {data_yaml}")

        train_config = dict(config)
        base_model = train_config.pop("base_model", None) or "yolo26s.pt"
        run_name = train_config.pop("run_name", f"train_{dataset_dir.name}")
        runs_root = Path(train_config.pop("runs_root", dataset_dir.parent / "runs")).resolve()
        if train_config.get("device") == "auto":
            train_config.pop("device")

        train_args = {
            "data": str(data_yaml),
            "project": str(runs_root),
            "name": run_name,
            "exist_ok": True,
            "save": True,
            **train_config,
        }

        model = YOLO(base_model)
        results = model.train(**train_args)
        save_dir = Path(getattr(results, "save_dir", runs_root / run_name))
        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"

        if not best_path.exists() and last_path.exists():
            best_path = last_path
        if not best_path.exists():
            raise HTTPException(status_code=500, detail=f"YOLO training did not produce weights in {save_dir}")

        metrics = self._extract_metrics(results)
        return {
            "metrics": metrics,
            "weights_path": best_path,
            "run_dir": save_dir,
        }

    def resume(self, checkpoint_path: Path) -> dict[str, Any]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="ultralytics is not installed. Run: pip install -e .",
            ) from exc

        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail="Training checkpoint not found")

        model = YOLO(str(checkpoint_path))
        results = model.train(resume=True)
        save_dir = Path(getattr(results, "save_dir", checkpoint_path.parents[1]))
        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"
        if not best_path.exists() and last_path.exists():
            best_path = last_path
        if not best_path.exists():
            raise HTTPException(status_code=500, detail=f"Resumed training did not produce weights in {save_dir}")
        return {
            "metrics": self._extract_metrics(results),
            "weights_path": best_path,
            "run_dir": save_dir,
        }

    def copy_weights_to(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)

    def _extract_metrics(self, results) -> dict[str, float | int | None]:
        results_dict = getattr(results, "results_dict", {}) or {}
        box = getattr(results, "box", None)
        pose = getattr(results, "pose", None)
        return {
            "epoch": None,
            "box_loss": self._metric(results_dict, "train/box_loss"),
            "cls_loss": self._metric(results_dict, "train/cls_loss"),
            "precision": self._metric(results_dict, "metrics/precision(B)") or self._metric(results_dict, "metrics/precision(P)") or getattr(box, "mp", None) or getattr(pose, "mp", None),
            "recall": self._metric(results_dict, "metrics/recall(B)") or self._metric(results_dict, "metrics/recall(P)") or getattr(box, "mr", None) or getattr(pose, "mr", None),
            "map50": self._metric(results_dict, "metrics/mAP50(B)") or self._metric(results_dict, "metrics/mAP50(P)") or getattr(box, "map50", None) or getattr(pose, "map50", None),
            "map50_95": self._metric(results_dict, "metrics/mAP50-95(B)") or self._metric(results_dict, "metrics/mAP50-95(P)") or getattr(box, "map", None) or getattr(pose, "map", None),
        }

    def _metric(self, values: dict, key: str):
        value = values.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
