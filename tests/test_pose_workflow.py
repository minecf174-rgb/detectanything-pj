from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.main import app


client = TestClient(app)


def register_pose_user() -> dict:
    response = client.post(
        "/auth/register",
        json={
            "email": "pose-workflow@example.com",
            "username": "pose-workflow",
            "password": "correct-horse-battery-staple",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_pretrained_pose_job_and_outside_frame_keypoints_are_supported():
    session = register_pose_user()
    headers = {"Authorization": f"Bearer {session['access_token']}"}
    project_response = client.post(
        "/projects",
        headers=headers,
        json={"name": "Pose workflow", "task_type": "POSE_ESTIMATION"},
    )
    assert project_response.status_code == 200, project_response.text
    project_id = project_response.json()["project_id"]

    class_response = client.post(
        "/classes",
        headers=headers,
        json={"project_id": project_id, "name": "person", "class_index": 0, "color": "#16a34a"},
    )
    assert class_response.status_code == 200, class_response.text
    class_id = class_response.json()["class_id"]

    image_bytes = BytesIO()
    PILImage.new("RGB", (64, 64), "white").save(image_bytes, format="PNG")
    upload_response = client.post(
        f"/images/upload?project_id={project_id}",
        headers=headers,
        files={"file": ("pose.png", image_bytes.getvalue(), "image/png")},
    )
    assert upload_response.status_code == 200, upload_response.text
    image_id = upload_response.json()["image_id"]

    job_response = client.post(
        "/auto-label-jobs",
        headers=headers,
        json={
            "project_id": project_id,
            "model_id": None,
            "base_model": "yolo26n-pose.pt",
            "image_ids": [image_id],
            "only_unlabeled": True,
        },
    )
    assert job_response.status_code == 200, job_response.text
    assert job_response.json()["model_id"] is None
    assert job_response.json()["base_model"] == "yolo26n-pose.pt"
    job_id = job_response.json()["job_id"]

    run_response = client.post(f"/auto-label-jobs/{job_id}/start-local-yolo", headers=headers)
    assert run_response.status_code == 200, run_response.text
    run_result = run_response.json()
    assert run_result["job"]["status"] == "COMPLETED"
    assert run_result["suggestions"][0]["source_model_id"] is None
    suggestion_id = run_result["suggestions"][0]["annotation_set_id"]

    retry_payload = {
        "project_id": project_id,
        "base_model": "yolo26n-pose.pt",
        "image_ids": [image_id],
        "only_unlabeled": True,
    }
    pending_retry = client.post("/auto-label-jobs", headers=headers, json=retry_payload)
    assert pending_retry.status_code == 400, pending_retry.text

    accept_response = client.post(
        "/annotations/sets/accept-bulk",
        headers=headers,
        json={"annotation_set_ids": [suggestion_id]},
    )
    assert accept_response.status_code == 200, accept_response.text
    assert accept_response.json() == {"accepted_count": 1, "skipped_count": 0}

    negative_dataset_response = client.post(
        "/datasets/versions",
        headers=headers,
        json={"project_id": project_id, "default_split": "TRAIN", "auto_split": True},
    )
    assert negative_dataset_response.status_code == 400, negative_dataset_response.text

    empty_retry = client.post("/auto-label-jobs", headers=headers, json=retry_payload)
    assert empty_retry.status_code == 200, empty_retry.text
    assert empty_retry.json()["total_images"] == 1
    assert [item["image_id"] for item in empty_retry.json()["items"]] == [image_id]

    keypoints = [
        {"x": 0.25 + index * 0.02, "y": 0.25 + index * 0.02, "visibility": 2}
        if index < 5 else {"x": 0, "y": 0, "visibility": 0}
        for index in range(17)
    ]
    annotation_response = client.post(
        f"/annotations/images/{image_id}/annotation-sets",
        headers=headers,
        json={
            "annotations": [{
                "class_id": class_id,
                "x_min": 0.2,
                "y_min": 0.2,
                "x_max": 0.4,
                "y_max": 0.4,
                "confidence": None,
                "keypoints": keypoints,
            }],
        },
    )
    assert annotation_response.status_code == 200, annotation_response.text
    saved_keypoints = annotation_response.json()["annotations"][0]["keypoints"]
    assert len(saved_keypoints) == 17
    assert saved_keypoints[-1]["visibility"] == 0

    labeled_retry = client.post("/auto-label-jobs", headers=headers, json=retry_payload)
    assert labeled_retry.status_code == 400, labeled_retry.text
    assert labeled_retry.json()["detail"] == "No unlabeled images available for auto-labeling"

    dataset_response = client.post(
        "/datasets/versions",
        headers=headers,
        json={"project_id": project_id, "default_split": "TRAIN", "auto_split": True},
    )
    assert dataset_response.status_code == 200, dataset_response.text
    version_id = dataset_response.json()["version_id"]
    build_response = client.post(f"/datasets/versions/{version_id}/build-yolo", headers=headers)
    assert build_response.status_code == 200, build_response.text
    label_file = next((Path(build_response.json()["output_dir"]) / "labels" / "train").glob("*.txt"))
    label_values = label_file.read_text().strip().split()
    assert label_values[-3:] == ["0.000000", "0.000000", "0"]
