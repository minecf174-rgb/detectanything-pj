from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.core.config.settings import get_settings
from app.main import app
from app.modules.images import previews
from app.modules.images.backfill_previews import main as backfill_previews


client = TestClient(app)


def _account(name: str) -> dict[str, str]:
    response = client.post("/auth/register", json={
        "email": f"preview-{name}@example.com",
        "username": f"preview-{name}",
        "password": "correct-horse-battery-staple",
    })
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _image_bytes() -> bytes:
    output = BytesIO()
    PILImage.new("RGB", (1600, 900), "#3070aa").save(output, format="PNG")
    return output.getvalue()


def test_preview_is_webp_cached_and_project_private(monkeypatch):
    owner = _account("owner")
    outsider = _account("outsider")
    project_response = client.post("/projects", headers=owner, json={"name": "Preview", "task_type": "OBJECT_DETECTION"})
    assert project_response.status_code == 200, project_response.text
    project_id = project_response.json()["project_id"]
    source = _image_bytes()
    upload = client.post(f"/images/upload?project_id={project_id}", headers=owner,
                         files={"file": ("source.png", source, "image/png")})
    assert upload.status_code == 200, upload.text
    image_id = upload.json()["image_id"]
    preview_files = list((Path(get_settings().storage_root) / "previews" / f"project_{project_id}").glob(f"image_{image_id}_*.webp"))
    assert len(preview_files) == 1  # Upload already scheduled and completed the preview.

    assert client.get(f"/images/{image_id}/preview").status_code == 401
    assert client.get(f"/images/{image_id}/preview", headers=outsider).status_code == 403
    response = client.get(f"/images/{image_id}/preview", headers=owner)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/webp"
    assert response.content[:4] == b"RIFF"
    with PILImage.open(BytesIO(response.content)) as preview:
        assert preview.format == "WEBP"
        assert preview.size == (768, 432)
    assert client.get(f"/images/{image_id}/preview", headers=owner).content == response.content
    assert client.get(f"/images/{image_id}/file", headers=owner).content == source

    preview_files[0].unlink()
    monkeypatch.setattr(sys, "argv", ["backfill_previews", "--project-id", str(project_id)])
    backfill_previews()
    assert preview_files[0].exists()
    backfill_previews()  # Safe to run again during setup.

    archived = client.post("/images/bulk-status", headers=owner, json={
        "project_id": project_id, "image_ids": [image_id], "status": "ARCHIVED",
    })
    assert archived.status_code == 200, archived.text
    deleted = client.post("/images/bulk-delete", headers=owner, json={"project_id": project_id, "image_ids": [image_id]})
    assert deleted.status_code == 200, deleted.text
    assert not preview_files[0].exists()


def test_both_zip_imports_generate_previews_without_changing_originals():
    owner = _account("zip-owner")
    project = client.post("/projects", headers=owner, json={"name": "Zip previews", "task_type": "OBJECT_DETECTION"}).json()
    project_id = project["project_id"]
    source = _image_bytes()
    plain_archive = BytesIO()
    with ZipFile(plain_archive, "w") as archive:
        archive.writestr("photos/one.png", source)
    plain = client.post(f"/images/import-image-archive?project_id={project_id}", headers=owner,
                        files={"file": ("plain.zip", plain_archive.getvalue(), "application/zip")})
    assert plain.status_code == 200, plain.text

    yolo_archive = BytesIO()
    with ZipFile(yolo_archive, "w") as archive:
        archive.writestr("data.yaml", "names: [person]\n")
        archive.writestr("images/train/two.png", source)
        archive.writestr("labels/train/two.txt", "0 0.5 0.5 0.5 0.5\n")
    yolo = client.post(f"/images/import-yolo?project_id={project_id}", headers=owner,
                       files={"file": ("yolo.zip", yolo_archive.getvalue(), "application/zip")})
    assert yolo.status_code == 200, yolo.text

    images = client.get(f"/images/project/{project_id}", headers=owner).json()
    assert len(images) == 2
    for image in images:
        image_id = image["image_id"]
        assert client.get(f"/images/{image_id}/file", headers=owner).content == source
        preview_files = list((Path(get_settings().storage_root) / "previews" / f"project_{project_id}").glob(f"image_{image_id}_*.webp"))
        assert len(preview_files) == 1


def test_preview_failure_does_not_fail_upload_and_can_be_recovered(monkeypatch):
    owner = _account("failed-preview")
    project = client.post("/projects", headers=owner, json={"name": "Recover preview", "task_type": "OBJECT_DETECTION"}).json()
    source = _image_bytes()

    def fail_preview(_source: Path, _destination: Path) -> Path:
        raise OSError("simulated thumbnail failure")

    with monkeypatch.context() as patch:
        patch.setattr(previews, "generate_preview", fail_preview)
        upload = client.post(f"/images/upload?project_id={project['project_id']}", headers=owner,
                             files={"file": ("source.png", source, "image/png")})
    assert upload.status_code == 200, upload.text
    image_id = upload.json()["image_id"]
    assert client.get(f"/images/{image_id}/file", headers=owner).content == source
    recovered = client.get(f"/images/{image_id}/preview", headers=owner)
    assert recovered.status_code == 200, recovered.text
    assert recovered.headers["content-type"] == "image/webp"


def test_annotation_preview_batch_returns_requested_sets_only_to_owner():
    owner = _account("batch-owner")
    outsider = _account("batch-outsider")
    project = client.post("/projects", headers=owner, json={"name": "Batch", "task_type": "OBJECT_DETECTION"}).json()
    project_id = project["project_id"]
    label = client.post("/classes", headers=owner, json={
        "project_id": project_id, "name": "person", "class_index": 0, "color": "#16a34a",
    }).json()
    image = client.post(f"/images/upload?project_id={project_id}", headers=owner,
                        files={"file": ("source.png", _image_bytes(), "image/png")}).json()
    annotation = client.post(f"/annotations/images/{image['image_id']}/annotation-sets", headers=owner,
                             json={"annotations": [{"class_id": label["class_id"], "x_min": 0.1,
                                                    "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}]}).json()
    set_id = annotation["annotation_set_id"]
    body = {"annotation_set_ids": [set_id]}
    assert client.post("/annotations/sets/previews", json=body).status_code == 401
    assert client.post("/annotations/sets/previews", headers=outsider, json=body).status_code == 404
    response = client.post("/annotations/sets/previews", headers=owner, json=body)
    assert response.status_code == 200, response.text
    assert response.json() == [annotation]
