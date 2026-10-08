from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.core.config.settings import get_settings
from app.main import app


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


def test_preview_is_webp_cached_and_project_private():
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

    preview_files = list((Path(get_settings().storage_root) / "previews" / f"project_{project_id}").glob(f"image_{image_id}_*.webp"))
    assert len(preview_files) == 1
    archived = client.post("/images/bulk-status", headers=owner, json={
        "project_id": project_id, "image_ids": [image_id], "status": "ARCHIVED",
    })
    assert archived.status_code == 200, archived.text
    deleted = client.post("/images/bulk-delete", headers=owner, json={"project_id": project_id, "image_ids": [image_id]})
    assert deleted.status_code == 200, deleted.text
    assert not preview_files[0].exists()


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
