from abc import ABC, abstractmethod
from pathlib import Path


class DatasetBuilder(ABC):
    @abstractmethod
    def build(self, dataset_version_id: int, output_dir: Path) -> Path:
        raise NotImplementedError
