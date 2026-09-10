from io import BytesIO
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.main import app


client = TestClient(app)


def _register(account: str) -> dict[str, str]:
    response = client.post(
        "/auth/register",
        json={
            "email": f"data-management-{account}@example.com",
            "username": f"data-management-{account}",
            "password": "correct-horse-battery-staple",
        },
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _png_bytes() -> bytes:
    output = BytesIO()
    PILImage.new("RGB", (32, 24), "white").save(output, format="PNG")
    return output.getvalue()


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_archive_inspection_and_plain_image_import():
    headers = _register("archive")
    project = client.post(
        "/projects",
        headers=headers,
        json={"name": "Archive import", "task_type": "OBJECT_DETECTION"},
    ).json()
    image_archive = _zip_bytes({"photos/one.png": _png_bytes(), "photos/two.png": _png_bytes()})

    inspection = client.post(
        "/images/inspect-archive",
        headers=headers,
        files={"file": ("images.zip", image_archive, "application/zip")},
    )
    assert inspection.status_code == 200, inspection.text
    assert inspection.json() == {"archive_type": "IMAGES", "image_count": 2, "class_names": []}

    imported = client.post(
        f"/images/import-image-archive?project_id={project['project_id']}",
        headers=headers,
        files={"file": ("images.zip", image_archive, "application/zip")},
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["imported_images"] == 2

    yolo_archive = _zip_bytes({
        "data.yaml": b"names: [person]\n",
        "images/train/one.png": _png_bytes(),
        "labels/train/one.txt": b"0 0.5 0.5 0.5 0.5\n",
    })
    yolo_inspection = client.post(
        "/images/inspect-archive",
        headers=headers,
        files={"file": ("dataset.zip", yolo_archive, "application/zip")},
    )
    assert yolo_inspection.status_code == 200, yolo_inspection.text
    assert yolo_inspection.json() == {
        "archive_type": "YOLO",
        "image_count": 1,
        "class_names": ["person"],
    }


def test_unused_dataset_version_can_be_deleted():
    headers = _register("dataset")
    project = client.post(
        "/projects",
        headers=headers,
        json={"name": "Dataset delete", "task_type": "OBJECT_DETECTION"},
    ).json()
    label_class = client.post(
        "/classes",
        headers=headers,
        json={"project_id": project["project_id"], "name": "object", "class_index": 0, "color": "#16a34a"},
    ).json()
    image = client.post(
        f"/images/upload?project_id={project['project_id']}",
        headers=headers,
        files={"file": ("one.png", _png_bytes(), "image/png")},
    ).json()
    annotation = client.post(
        f"/annotations/images/{image['image_id']}/annotation-sets",
        headers=headers,
        json={"annotations": [{
            "class_id": label_class["class_id"],
            "x_min": 0.1,
            "y_min": 0.1,
            "x_max": 0.9,
            "y_max": 0.9,
            "confidence": None,
        }]},
    )
    assert annotation.status_code == 200, annotation.text
    statuses = client.get(
        f"/annotations/projects/{project['project_id']}/image-statuses",
        headers=headers,
    )
    assert statuses.status_code == 200, statuses.text
    assert statuses.json() == [{"image_id": image["image_id"], "status": "APPROVED", "box_count": 1}]
    version = client.post(
        "/datasets/versions",
        headers=headers,
        json={"project_id": project["project_id"], "default_split": "TRAIN", "auto_split": True},
    ).json()

    deleted = client.delete(f"/datasets/versions/{version['version_id']}", headers=headers)
    assert deleted.status_code == 204, deleted.text
    assert client.get(f"/datasets/projects/{project['project_id']}/versions", headers=headers).json() == []
