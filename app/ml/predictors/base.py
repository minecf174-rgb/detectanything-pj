from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Predictor(ABC):
    @abstractmethod
    def predict(self, model_path: Path, image_paths: list[Path], config: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

