"""Generated, disposable WebP previews for image browsing."""

from hashlib import sha256
import logging
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Semaphore

from PIL import Image as PILImage
from PIL import ImageOps


logger = logging.getLogger(__name__)
_conversion_slots = Semaphore(2)


def preview_path(
    storage_root: Path,
    project_id: int,
    image_id: int,
    object_key: str,
    file_hash: str,
) -> Path:
    version = sha256(f"{object_key}:{file_hash}:webp-768-v1".encode()).hexdigest()[:20]
    return storage_root / "previews" / f"project_{project_id}" / f"image_{image_id}_{version}.webp"


def generate_preview(source_path: Path, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=destination.parent, suffix=".webp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with PILImage.open(source_path) as source:
            preview = ImageOps.exif_transpose(source)
            preview.thumbnail((768, 768), PILImage.Resampling.LANCZOS)
            if preview.mode not in {"RGB", "RGBA"}:
                has_alpha = "A" in preview.getbands() or "transparency" in preview.info
                preview = preview.convert("RGBA" if has_alpha else "RGB")
            preview.save(temporary_path, format="WEBP", quality=78, method=4)
        replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def prepare_previews(paths: list[tuple[Path, Path]]) -> None:
    """Run after the upload response; a failed preview never removes its original."""
    for source_path, destination in paths:
        try:
            with _conversion_slots:
                generate_preview(source_path, destination)
        except Exception:
            logger.exception("Could not generate image preview: %s", source_path)
