import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.core.config.settings import get_settings
from app.core.database.session import SessionLocal
from app.main import app
from app.ml.predictors.ultralytics_yolo import UltralyticsYoloPredictor
from app.ml.trainers.ultralytics_yolo import UltralyticsYoloTrainer
from app.modules.models.service import ModelService
from app.modules.training.models import TrainingRun
from app.modules.training.service import TrainingService

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


def _pt_upload() -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"test")
    return output.getvalue()


def test_import_model_is_project_scoped_and_imports_classes(monkeypatch):
    owner = _register("import-owner")
    other = _register("import-other")
    project = client.post(
        "/projects", headers=owner,
        json={"name": "Imported model", "task_type": "OBJECT_DETECTION"},
    ).json()
    other_project = client.post(
        "/projects", headers=owner,
        json={"name": "Other project", "task_type": "OBJECT_DETECTION"},
    ).json()
    monkeypatch.setattr(ModelService, "_inspect_uploaded_weights", staticmethod(lambda _path: ("detect", ["person", "car"], b"safe weights")))
    payload = _pt_upload()

    blocked = client.post(
        f"/models/import?project_id={project['project_id']}", headers=other,
        files={"file": ("best.pt", payload, "application/octet-stream")},
    )
    assert blocked.status_code == 403
    imported = client.post(
        f"/models/import?project_id={project['project_id']}", headers=owner,
        files={"file": ("best.pt", payload, "application/octet-stream")},
    )
    assert imported.status_code == 201, imported.text
    model = imported.json()["model"]
    assert model["source_type"] == "IMPORTED"
    assert model["training_run_id"] is None
    assert model["class_names"] == ["person", "car"]
    assert imported.json()["created_classes"] == ["person", "car"]
    project_classes = client.get(f"/classes/project/{project['project_id']}", headers=owner).json()
    assert [(item["class_index"], item["name"]) for item in project_classes] == [(0, "person"), (1, "car")]
    assert client.delete(f"/classes/{project_classes[0]['class_id']}", headers=owner).status_code == 409
    assert client.patch(
        f"/classes/{project_classes[0]['class_id']}", headers=owner,
        json={"name": "renamed", "color": "#16a34a"},
    ).status_code == 409
    assert client.get(f"/models/project/{other_project['project_id']}", headers=owner).json() == []
    assert client.get(f"/models/project/{project['project_id']}", headers=other).status_code == 403
    assert client.get(f"/models/{model['model_id']}", headers=other).status_code == 403
    assert client.get(f"/models/{model['model_id']}/download", headers=other).status_code == 403
    other_image = client.post(
        f"/images/upload?project_id={other_project['project_id']}", headers=owner,
        files={"file": ("one.png", _png_bytes(), "image/png")},
    ).json()
    cross_project_job = client.post(
        "/auto-label-jobs", headers=owner,
        json={"project_id": other_project["project_id"], "model_id": model["model_id"],
              "image_ids": [other_image["image_id"]], "only_unlabeled": False},
    )
    assert cross_project_job.status_code == 400

    repeated = client.post(
        f"/models/import?project_id={project['project_id']}", headers=owner,
        files={"file": ("best.pt", payload, "application/octet-stream")},
    )
    assert repeated.status_code == 201, repeated.text
    assert repeated.json()["created_classes"] == []
    assert repeated.json()["model"]["version"] == 2
    weights_path = Path(get_settings().storage_root) / model["weights_object_key"]
    assert weights_path.exists()
    assert client.delete(f"/projects/{project['project_id']}", headers=owner).status_code == 204
    assert not weights_path.exists()


def test_import_model_rejects_class_or_task_mismatch_without_changes(monkeypatch):
    owner = _register("import-conflict")
    detect_project = client.post(
        "/projects", headers=owner,
        json={"name": "Conflict", "task_type": "OBJECT_DETECTION"},
    ).json()
    pose_project = client.post(
        "/projects", headers=owner,
        json={"name": "Pose conflict", "task_type": "POSE_ESTIMATION"},
    ).json()
    client.post(
        "/classes", headers=owner,
        json={"project_id": detect_project["project_id"], "name": "object", "class_index": 0, "color": "#16a34a"},
    )
    monkeypatch.setattr(ModelService, "_inspect_uploaded_weights", staticmethod(lambda _path: ("detect", ["person"], b"safe weights")))
    payload = _pt_upload()
    conflict = client.post(
        f"/models/import?project_id={detect_project['project_id']}", headers=owner,
        files={"file": ("best.pt", payload, "application/octet-stream")},
    )
    assert conflict.status_code == 409
    assert "Class 0" in conflict.text
    wrong_task = client.post(
        f"/models/import?project_id={pose_project['project_id']}", headers=owner,
        files={"file": ("best.pt", payload, "application/octet-stream")},
    )
    assert wrong_task.status_code == 400
    assert client.get(f"/models/project/{detect_project['project_id']}", headers=owner).json() == []
    assert client.get(f"/models/project/{pose_project['project_id']}", headers=owner).json() == []


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


def test_interrupted_training_can_resume_from_last_checkpoint(monkeypatch, tmp_path):
    headers = _register("resume-owner")
    project_id, version_id = _create_training_fixture(headers)
    created = client.post(
        "/training-runs",
        headers=headers,
        json={
            "project_id": project_id,
            "version_id": version_id,
            "model_type": "YOLO26_DETECT",
            "base_model": "yolo26s.pt",
        },
    )
    assert created.status_code == 200, created.text
    run_id = created.json()["training_run_id"]

    with SessionLocal() as db:
        run = db.get(TrainingRun, run_id)
        assert run is not None
        run.status = "RUNNING"
        db.commit()
        checkpoint = TrainingService(db)._checkpoint_path(run)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        assert TrainingService.recover_interrupted_runs(db) == 1
        db.refresh(run)
        assert run.status == "INTERRUPTED"

    resumed_weights = tmp_path / "best.pt"
    resumed_weights.write_bytes(b"resumed weights")

    def fake_resume(_self, checkpoint_path):
        assert checkpoint_path.name == "last.pt"
        return {
            "metrics": {
                "epoch": 2,
                "box_loss": 0.1,
                "cls_loss": 0.1,
                "precision": 0.8,
                "recall": 0.7,
                "map50": 0.75,
                "map50_95": 0.5,
            },
            "weights_path": resumed_weights,
            "run_dir": checkpoint_path.parents[1],
        }

    monkeypatch.setattr(UltralyticsYoloTrainer, "resume", fake_resume)

    listed = client.get(f"/training-runs/project/{project_id}", headers=headers)
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["can_resume"] is True

    resumed = client.post(f"/training-runs/{run_id}/resume-local-yolo", headers=headers)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["training_run"]["status"] == "COMPLETED"
    assert resumed.json()["model"]["version"] == 1
    assert not checkpoint.parent.parent.exists()
    assert not (Path(get_settings().temp_root) / "training" / f"dataset_version_{version_id}").exists()


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
