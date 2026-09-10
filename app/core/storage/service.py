from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree


class StorageService(ABC):
    @abstractmethod
    def save(self, object_key: str, data: bytes) -> str:
        raise NotImplementedError

    @abstractmethod
    def get(self, object_key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def delete(self, object_key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, object_key: str) -> bool:
        raise NotImplementedError


class LocalStorageService(StorageService):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, object_key: str, data: bytes) -> str:
        path = self.root / object_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return object_key

    def get(self, object_key: str) -> bytes:
        return (self.root / object_key).read_bytes()

    def delete(self, object_key: str) -> None:
        path = self.root / object_key
        if path.exists():
            path.unlink()

    def exists(self, object_key: str) -> bool:
        return (self.root / object_key).exists()

    def delete_prefix(self, object_key_prefix: str) -> None:
        path = (self.root / object_key_prefix).resolve()
        root = self.root.resolve()
        if path == root or root not in path.parents:
            raise ValueError("Storage prefix must be inside the storage root")
        if path.is_dir():
            rmtree(path)
        elif path.exists():
            path.unlink()
