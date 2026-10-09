from io import BytesIO
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.core.database.session import SessionLocal
from app.main import app
from app.modules.auto_label.repository import AutoLabelRepository


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
    assert inspection.json() == {
        "archive_type": "IMAGES",
        "image_count": 2,
        "labeled_image_count": 0,
        "unlabeled_image_count": 2,
        "class_names": [],
    }

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
        "labeled_image_count": 1,
        "unlabeled_image_count": 0,
        "class_names": ["person"],
    }


def test_yolo_label_limit_can_keep_or_skip_remaining_images():
    headers = _register("yolo-label-limit")
    project = client.post(
        "/projects",
        headers=headers,
        json={"name": "YOLO label limit", "task_type": "OBJECT_DETECTION"},
    ).json()
    yolo_archive = _zip_bytes({
        "data.yaml": b"names: [person]\n",
        "images/train/one.png": _png_bytes(),
        "images/train/two.png": _png_bytes(),
        "images/train/three.png": _png_bytes(),
        "labels/train/one.txt": b"0 0.5 0.5 0.5 0.5\n",
        "labels/train/two.txt": b"0 0.4 0.4 0.2 0.2\n",
    })

    inspection = client.post(
        "/images/inspect-archive",
        headers=headers,
        files={"file": ("dataset.zip", yolo_archive, "application/zip")},
    )
    assert inspection.status_code == 200, inspection.text
    assert inspection.json()["labeled_image_count"] == 2
    assert inspection.json()["unlabeled_image_count"] == 1

    keep_remaining = client.post(
        f"/images/import-yolo?project_id={project['project_id']}&max_labeled_images=1&import_remaining_images=true",
        headers=headers,
        files={"file": ("dataset.zip", yolo_archive, "application/zip")},
    )
    assert keep_remaining.status_code == 200, keep_remaining.text
    assert keep_remaining.json()["imported_images"] == 3
    assert keep_remaining.json()["imported_annotation_sets"] == 1

    skip_remaining = client.post(
        f"/images/import-yolo?project_id={project['project_id']}&max_labeled_images=1&import_remaining_images=false",
        headers=headers,
        files={"file": ("dataset.zip", yolo_archive, "application/zip")},
    )
    assert skip_remaining.status_code == 200, skip_remaining.text
    assert skip_remaining.json()["imported_images"] == 1
    assert skip_remaining.json()["imported_annotation_sets"] == 1


def test_yolo_pose_zip_imports_box_and_keypoints():
    headers = _register("yolo-pose-import")
    project = client.post(
        "/projects",
        headers=headers,
        json={"name": "YOLO pose import", "task_type": "POSE_ESTIMATION"},
    ).json()
    keypoints = " ".join(
        f"{0.20 + index * 0.01:.2f} {0.30 + index * 0.01:.2f} {0 if index == 16 else 2}"
        for index in range(17)
    )
    yolo_archive = _zip_bytes({
        "data.yaml": b"names: [person]\nkpt_shape: [17, 3]\n",
        "images/train/pose.png": _png_bytes(),
        "labels/train/pose.txt": f"0 0.5 0.5 0.4 0.6 {keypoints}\n".encode(),
    })

    imported = client.post(
        f"/images/import-yolo?project_id={project['project_id']}",
        headers=headers,
        files={"file": ("pose-dataset.zip", yolo_archive, "application/zip")},
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["imported_images"] == 1
    assert imported.json()["imported_annotation_sets"] == 1
    assert imported.json()["imported_boxes"] == 1

    images = client.get(f"/images/project/{project['project_id']}", headers=headers).json()
    annotation_sets = client.get(
        f"/annotations/images/{images[0]['image_id']}/annotation-sets",
        headers=headers,
    )
    assert annotation_sets.status_code == 200, annotation_sets.text
    annotation = annotation_sets.json()[0]["annotations"][0]
    assert annotation["x_min"] == 0.3
    assert annotation["y_min"] == 0.2
    assert annotation["x_max"] == 0.7
    assert annotation["y_max"] == 0.8
    assert len(annotation["keypoints"]) == 17
    assert annotation["keypoints"][0] == {
        "keypoint_index": 0,
        "x": 0.2,
        "y": 0.3,
        "visibility": 2,
    }
    assert annotation["keypoints"][16]["visibility"] == 0


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
    assert statuses.json() == [{
        "image_id": image["image_id"],
        "annotation_set_id": annotation.json()["annotation_set_id"],
        "source_type": "HUMAN",
        "status": "APPROVED",
        "box_count": 1,
        "is_negative": False,
        "class_ids": [label_class["class_id"]],
    }]
    version = client.post(
        "/datasets/versions",
        headers=headers,
        json={"project_id": project["project_id"], "default_split": "TRAIN", "auto_split": True},
    ).json()

    deleted = client.delete(f"/datasets/versions/{version['version_id']}", headers=headers)
    assert deleted.status_code == 204, deleted.text
    assert client.get(f"/datasets/projects/{project['project_id']}/versions", headers=headers).json() == []


def test_dataset_version_excludes_approved_images_without_labels():
    headers = _register("dataset-no-empty-labels")
    project = client.post(
        "/projects",
        headers=headers,
        json={"name": "Labeled dataset only", "task_type": "OBJECT_DETECTION"},
    ).json()
    label_class = client.post(
        "/classes",
        headers=headers,
        json={"project_id": project["project_id"], "name": "object", "class_index": 0, "color": "#16a34a"},
    ).json()
    labeled_image = client.post(
        f"/images/upload?project_id={project['project_id']}",
        headers=headers,
        files={"file": ("labeled.png", _png_bytes(), "image/png")},
    ).json()
    empty_image = client.post(
        f"/images/upload?project_id={project['project_id']}",
        headers=headers,
        files={"file": ("empty.png", _png_bytes(), "image/png")},
    ).json()
    client.post(
        f"/annotations/images/{labeled_image['image_id']}/annotation-sets",
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
    empty_annotation = client.post(
        f"/annotations/images/{empty_image['image_id']}/annotation-sets",
        headers=headers,
        json={"annotations": []},
    )
    assert empty_annotation.status_code == 200, empty_annotation.text

    version = client.post(
        "/datasets/versions",
        headers=headers,
        json={"project_id": project["project_id"], "default_split": "TRAIN", "auto_split": True},
    )
    assert version.status_code == 200, version.text
    assert [item["image_id"] for item in version.json()["items"]] == [labeled_image["image_id"]]

    invalid_selection = client.post(
        "/datasets/versions",
        headers=headers,
        json={"project_id": project["project_id"], "image_ids": [empty_image["image_id"]]},
    )
    assert invalid_selection.status_code == 400, invalid_selection.text

    empty_selection = client.post(
        "/datasets/versions",
        headers=headers,
        json={"project_id": project["project_id"], "image_ids": []},
    )
    assert empty_selection.status_code == 422, empty_selection.text

    added_label = client.post(
        f"/annotations/images/{empty_image['image_id']}/annotation-sets",
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
    assert added_label.status_code == 200, added_label.text
    subset = client.post(
        "/datasets/versions",
        headers=headers,
        json={
            "project_id": project["project_id"],
            "image_ids": [empty_image["image_id"], empty_image["image_id"]],
        },
    )
    assert subset.status_code == 200, subset.text
    assert [item["image_id"] for item in subset.json()["items"]] == [empty_image["image_id"]]
    assert subset.json()["items"][0]["annotation_set_id"] == added_label.json()["annotation_set_id"]
    marked_negative = client.post(
        f"/annotations/images/{empty_image['image_id']}/confirm-negative", headers=headers,
    )
    assert marked_negative.status_code == 200, marked_negative.text
    assert marked_negative.json()["is_negative"] is True
    assert marked_negative.json()["annotations"] == []
    history = client.get(f"/annotations/images/{empty_image['image_id']}/annotation-sets", headers=headers)
    assert history.status_code == 200, history.text
    assert len(history.json()[-2]["annotations"]) == 1
    assert history.json()[-1]["is_negative"] is True


def test_verified_negative_enters_dataset_with_empty_yolo_label():
    headers = _register("verified-negative")
    project = client.post(
        "/projects", headers=headers,
        json={"name": "Verified negatives", "task_type": "OBJECT_DETECTION"},
    ).json()
    client.post(
        "/classes", headers=headers,
        json={"project_id": project["project_id"], "name": "object", "class_index": 0, "color": "#16a34a"},
    )
    image = client.post(
        f"/images/upload?project_id={project['project_id']}", headers=headers,
        files={"file": ("background.png", _png_bytes(), "image/png")},
    ).json()
    image_id = image["image_id"]
    unverified = client.post(
        f"/annotations/images/{image_id}/annotation-sets", headers=headers,
        json={"annotations": []},
    )
    assert unverified.status_code == 200, unverified.text
    assert unverified.json()["is_negative"] is False
    excluded = client.post(
        "/datasets/versions", headers=headers,
        json={"project_id": project["project_id"], "default_split": "TRAIN"},
    )
    assert excluded.status_code == 400, excluded.text

    confirmed = client.post(f"/annotations/images/{image_id}/confirm-negative", headers=headers)
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["is_negative"] is True
    assert confirmed.json()["annotations"] == []
    confirmed_again = client.post(f"/annotations/images/{image_id}/confirm-negative", headers=headers)
    assert confirmed_again.json()["annotation_set_id"] == confirmed.json()["annotation_set_id"]
    statuses = client.get(f"/annotations/projects/{project['project_id']}/image-statuses", headers=headers)
    assert statuses.json()[0]["is_negative"] is True

    version = client.post(
        "/datasets/versions", headers=headers,
        json={"project_id": project["project_id"], "image_ids": [image_id], "default_split": "TRAIN"},
    )
    assert version.status_code == 200, version.text
    assert version.json()["items"][0]["annotation_set_id"] == confirmed.json()["annotation_set_id"]
    export = client.get(f"/datasets/versions/{version.json()['version_id']}/export-yolo/download", headers=headers)
    assert export.status_code == 200, export.text
    with ZipFile(BytesIO(export.content)) as archive:
        label_files = [name for name in archive.namelist() if name.startswith("labels/") and name.endswith(".txt")]
        assert len(label_files) == 1
        assert archive.read(label_files[0]) == b""


def test_verifying_negative_clears_pending_boxes_without_erasing_history():
    headers = _register("negative-with-boxes")
    project = client.post(
        "/projects", headers=headers,
        json={"name": "Negative with boxes", "task_type": "OBJECT_DETECTION"},
    ).json()
    label_class = client.post(
        "/classes", headers=headers,
        json={"project_id": project["project_id"], "name": "object", "class_index": 0, "color": "#16a34a"},
    ).json()
    image = client.post(
        f"/images/upload?project_id={project['project_id']}", headers=headers,
        files={"file": ("false-positive.png", _png_bytes(), "image/png")},
    ).json()
    image_id = image["image_id"]
    with SessionLocal() as db:
        suggestions = [AutoLabelRepository(db).create_model_suggestion(
            image_id=image_id,
            model_id=None,
            boxes=[{
                "class_id": label_class["class_id"],
                "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9,
                "confidence": 0.8,
            }],
        ) for _ in range(2)]
        pending_ids = [item.annotation_set_id for item in suggestions]
        db.commit()

    confirmed = client.post(f"/annotations/images/{image_id}/confirm-negative", headers=headers)
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["is_negative"] is True
    assert confirmed.json()["annotations"] == []

    history = client.get(f"/annotations/images/{image_id}/annotation-sets", headers=headers)
    assert history.status_code == 200, history.text
    assert [item["status"] for item in history.json() if item["annotation_set_id"] in pending_ids] == [
        "REJECTED", "REJECTED",
    ]
    assert all(item["annotations"] for item in history.json() if item["annotation_set_id"] in pending_ids)
    assert history.json()[-1]["annotation_set_id"] == confirmed.json()["annotation_set_id"]


def test_image_status_class_ids_include_active_multilabel_and_ignore_rejected_suggestions():
    headers = _register("class-filter-status")
    project_id = client.post("/projects", headers=headers, json={
        "name": "Class filter", "task_type": "OBJECT_DETECTION",
    }).json()["project_id"]
    class_ids = [client.post("/classes", headers=headers, json={
        "project_id": project_id, "name": name, "class_index": index, "color": "#16a34a",
    }).json()["class_id"] for index, name in enumerate(("bottle", "cap"))]
    image_id = client.post(f"/images/upload?project_id={project_id}", headers=headers,
                           files={"file": ("items.png", _png_bytes(), "image/png")}).json()["image_id"]
    approved = client.post(f"/annotations/images/{image_id}/annotation-sets", headers=headers,
                           json={"annotations": [{"class_id": class_ids[0], "x_min": .1, "y_min": .1,
                                                   "x_max": .5, "y_max": .5} ]})
    assert approved.status_code == 200, approved.text
    with SessionLocal() as db:
        suggestion = AutoLabelRepository(db).create_model_suggestion(image_id=image_id, model_id=None,
            boxes=[{"class_id": class_ids[1], "x_min": .5, "y_min": .5,
                    "x_max": .9, "y_max": .9, "confidence": .8}])
        suggestion_id = suggestion.annotation_set_id
        db.commit()

    url = f"/annotations/projects/{project_id}/image-statuses"
    current = client.get(url, headers=headers).json()[0]
    assert current["status"] == "PENDING"
    assert current["class_ids"] == class_ids

    rejected = client.post(f"/annotations/sets/{suggestion_id}/reject", headers=headers, json={})
    assert rejected.status_code == 200, rejected.text
    current = client.get(url, headers=headers).json()[0]
    assert current["class_ids"] == [class_ids[0]]

    negative = client.post(f"/annotations/images/{image_id}/confirm-negative", headers=headers)
    assert negative.status_code == 200, negative.text
    assert client.get(url, headers=headers).json()[0]["class_ids"] == []


def test_class_name_and_color_can_be_updated_without_changing_index():
    owner_headers = _register("class-owner")
    other_headers = _register("class-other")
    project = client.post(
        "/projects",
        headers=owner_headers,
        json={"name": "Editable classes", "task_type": "OBJECT_DETECTION"},
    ).json()
    label_class = client.post(
        "/classes",
        headers=owner_headers,
        json={
            "project_id": project["project_id"],
            "name": "old-name",
            "class_index": 0,
            "color": "#16a34a",
        },
    ).json()

    forbidden = client.patch(
        f"/classes/{label_class['class_id']}",
        headers=other_headers,
        json={"name": "not-allowed", "color": "#e11d48"},
    )
    assert forbidden.status_code == 403, forbidden.text

    updated = client.patch(
        f"/classes/{label_class['class_id']}",
        headers=owner_headers,
        json={"name": "renamed-object", "color": "#2563EB"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "renamed-object"
    assert updated.json()["color"] == "#2563eb"
    assert updated.json()["class_index"] == 0
