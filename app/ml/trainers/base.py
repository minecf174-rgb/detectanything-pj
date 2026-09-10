from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Trainer(ABC):
    @abstractmethod
    def train(self, dataset_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

