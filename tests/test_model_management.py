import json
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.core.config.settings import get_settings
from app.main import app
from app.ml.predictors.ultralytics_yolo import UltralyticsYoloPredictor

client = TestClient(app)


def _register(account: str) -> dict[str, str]:
    response = client.post(
        "/auth/register",
        json={
            "email": f"model-management-{account}@example.com",
            "username": f"model-management-{account}",
            "password": "correct-horse-battery-staple",
        },
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _png_bytes() -> bytes:
    output = BytesIO()
    PILImage.new("RGB", (32, 24), "white").save(output, format="PNG")
    return output.getvalue()


def _create_training_fixture(headers: dict[str, str]) -> tuple[int, int]:
    project = client.post(
        "/projects",
        headers=headers,
        json={"name": "Model deletion", "task_type": "OBJECT_DETECTION"},
    ).json()
    project_id = project["project_id"]
    label_class = client.post(
        "/classes",
        headers=headers,
        json={"project_id": project_id, "name": "object", "class_index": 0, "color": "#16a34a"},
    ).json()
    image = client.post(
        f"/images/upload?project_id={project_id}",
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
    dataset = client.post(
        "/datasets/versions",
        headers=headers,
        json={"project_id": project_id, "default_split": "TRAIN", "auto_split": True},
    ).json()
    return project_id, dataset["version_id"]


def _train_mock(headers: dict[str, str], project_id: int, version_id: int) -> dict:
    run = client.post(
        "/training-runs",
        headers=headers,
        json={
            "project_id": project_id,
            "version_id": version_id,
            "model_type": "YOLO26_DETECT",
            "base_model": "yolo26s.pt",
        },
    )
    assert run.status_code == 200, run.text
    trained = client.post(
        f"/training-runs/{run.json()['training_run_id']}/start-local-mock",
        headers=headers,
    )
    assert trained.status_code == 200, trained.text
    return trained.json()["model"]


def test_model_files_can_be_deleted_without_reusing_the_version_number(monkeypatch):
    owner_headers = _register("owner")
    other_headers = _register("other")
    project_id, version_id = _create_training_fixture(owner_headers)
    first_model = _train_mock(owner_headers, project_id, version_id)
    model_id = first_model["model_id"]
    weights_path = Path(get_settings().storage_root) / first_model["weights_object_key"]
    assert first_model["version"] == 1
    assert weights_path.exists()

    def fake_prediction_stream(_self, _model_path, image_paths, _config):
        for image_path in image_paths:
            yield {"image_path": image_path, "boxes": []}

    monkeypatch.setattr(UltralyticsYoloPredictor, "predict_stream", fake_prediction_stream)
    with client.stream(
        "POST",
        f"/models/{model_id}/predict-batch-stream?confidence_threshold=0.25",
        headers=owner_headers,
        files=[
            ("files", ("test-1.png", _png_bytes(), "image/png")),
            ("files", ("test-2.png", _png_bytes(), "image/png")),
        ],
    ) as streamed:
        assert streamed.status_code == 200, streamed.text
        events = [json.loads(line) for line in streamed.iter_lines() if line]
    assert [event["index"] for event in events] == [0, 1]
    assert all(event["result"]["boxes"] == [] for event in events)

    image_id = client.get(f"/images/project/{project_id}", headers=owner_headers).json()[0]["image_id"]
    queued_job = client.post(
        "/auto-label-jobs",
        headers=owner_headers,
        json={
            "project_id": project_id,
            "model_id": model_id,
            "image_ids": [image_id],
            "only_unlabeled": False,
        },
    )
    assert queued_job.status_code == 200, queued_job.text
    blocked = client.delete(f"/models/{model_id}", headers=owner_headers)
    assert blocked.status_code == 409, blocked.text

    completed_job = client.post(
        f"/auto-label-jobs/{queued_job.json()['job_id']}/start-local-mock",
        headers=owner_headers,
    )
    assert completed_job.status_code == 200, completed_job.text
    assert client.delete(f"/models/{model_id}", headers=other_headers).status_code == 403

    deleted = client.delete(f"/models/{model_id}", headers=owner_headers)
    assert deleted.status_code == 204, deleted.text
    assert not weights_path.parent.exists()
    assert client.get(f"/models/{model_id}", headers=owner_headers).status_code == 404
    assert client.get(f"/models/project/{project_id}", headers=owner_headers).json() == []

    second_model = _train_mock(owner_headers, project_id, version_id)
    assert second_model["version"] == 2
