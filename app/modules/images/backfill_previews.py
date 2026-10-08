"""Generate missing WebP previews for images imported before upload-time previews."""

import argparse
import logging
from pathlib import Path

from sqlalchemy import select

from app.core.config.settings import get_settings
from app.core.database.session import SessionLocal
from app.modules.images.models import Image
from app.modules.images.previews import generate_preview, preview_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build missing image-browsing WebP previews")
    parser.add_argument("--project-id", type=int, help="Only backfill one project")
    args = parser.parse_args()
    storage_root = Path(get_settings().storage_root)
    generated = skipped = failed = processed = 0
    query = select(Image).where(Image.status != "ARCHIVED").order_by(Image.image_id)
    if args.project_id is not None:
        query = query.where(Image.project_id == args.project_id)
    with SessionLocal() as db:
        for image in db.scalars(query.execution_options(yield_per=100)):
            processed += 1
            destination = preview_path(storage_root, image.project_id, image.image_id, image.object_key, image.file_hash)
            if destination.exists():
                skipped += 1
            else:
                try:
                    generate_preview(storage_root / image.object_key, destination)
                    generated += 1
                except Exception:
                    logging.exception("Could not backfill preview for image %s", image.image_id)
                    failed += 1
            if processed % 100 == 0:
                print(f"Previews checked: {processed} (generated={generated}, failed={failed})", flush=True)
    print(f"Previews: generated={generated}, already_present={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
