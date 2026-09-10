from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.images.models import Image


class ImageRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        *,
        project_id: int,
        original_filename: str | None,
        object_key: str,
        file_hash: str,
        width: int | None,
        height: int | None,
        status: str,
    ) -> Image:
        image = Image(
            project_id=project_id,
            original_filename=original_filename,
            object_key=object_key,
            file_hash=file_hash,
            width=width,
            height=height,
            status=status,
        )
        self.db.add(image)
        self.db.flush()
        return image

    def get(self, image_id: int) -> Image | None:
        return self.db.get(Image, image_id)

    def list_by_project(self, project_id: int, include_archived: bool = False) -> list[Image]:
        query = select(Image).where(Image.project_id == project_id)
        if not include_archived:
            query = query.where(Image.status != "ARCHIVED")
        return list(
            self.db.scalars(query.order_by(Image.image_id))
        )
