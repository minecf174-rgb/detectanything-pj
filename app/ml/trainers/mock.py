from pathlib import Path
from typing import Any

from app.ml.trainers.base import Trainer


class MockYoloTrainer(Trainer):
    def train(self, dataset_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
        epochs = int(config.get("epochs", 1))
        metrics = {
            "epoch": epochs,
            "box_loss": 0.12,
            "cls_loss": 0.08,
            "precision": 0.91,
            "recall": 0.87,
            "map50": 0.89,
            "map50_95": 0.62,
        }
        weights = (
            "mock yolo weights\n"
            f"dataset_dir={dataset_dir}\n"
            f"config={config}\n"
            f"metrics={metrics}\n"
        ).encode("utf-8")
        return {"metrics": metrics, "weights": weights}
