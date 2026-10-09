from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.core.config.settings import get_settings
from app.core.database.session import SessionLocal
from app.main import app
from app.modules.auto_label.models import AutoLabelJob
from app.modules.datasets.models import DatasetVersion
from app.modules.models.models import Model
from app.modules.projects.models import Project
from app.modules.training.models import TrainingRun
from app.modules.users.service import UserService


client = TestClient(app)


def _register(name: str) -> dict[str, str]:
    response = client.post("/auth/register", json={
        "email": f"project-cleanup-{name}@example.com",
        "username": f"project-cleanup-{name}",
        "password": "correct-horse-battery-staple",
    })
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _image_bytes() -> bytes:
    output = BytesIO()
    PILImage.new("RGB", (80, 60), "#2255aa").save(output, format="PNG")
    return output.getvalue()


def _make_project_artifacts(headers: dict[str, str], name: str) -> tuple[int, list[Path], int, int]:
    project = client.post("/projects", headers=headers, json={
        "name": name,
        "task_type": "OBJECT_DETECTION",
    }).json()
    project_id = project["project_id"]
    upload = client.post(f"/images/upload?project_id={project_id}", headers=headers,
                         files={"file": ("original.png", _image_bytes(), "image/png")})
    assert upload.status_code == 200, upload.text
    image_id = upload.json()["image_id"]

    with SessionLocal() as db:
        version = DatasetVersion(project_id=project_id, version_number=1)
        db.add(version)
        db.flush()
        run = TrainingRun(project_id=project_id, version_id=version.version_id,
                          model_type="YOLO26_DETECT", base_model="yolo26n.pt",
                          status="COMPLETED")
        db.add(run)
        db.flush()
        model = Model(project_id=project_id, training_run_id=run.training_run_id,
                      source_type="TRAINED", model_type="YOLO26_DETECT", version=1,
                      weights_object_key=f"models/project_{project_id}/model_1/best.pt")
        db.add(model)
        db.commit()
        version_id, run_id = version.version_id, run.training_run_id

    settings = get_settings()
    storage = Path(settings.storage_root)
    temp = Path(settings.temp_root)
    original = next((storage / "projects" / f"project_{project_id}" / "images").iterdir())
    preview = next((storage / "previews" / f"project_{project_id}").glob("*.webp"))
    weights = storage / "models" / f"project_{project_id}" / "model_1" / "best.pt"
    built = temp / "training" / f"dataset_version_{version_id}" / "data.yaml"
    exported = temp / "training" / f"dataset_version_{version_id}.zip"
    checkpoint = temp / "training" / "runs" / f"training_run_{run_id}" / "weights" / "last.pt"
    for path in (weights, built, exported, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    return project_id, [original, preview, weights, built, exported, checkpoint], version_id, run_id


def test_project_delete_removes_only_its_storage_and_training_artifacts():
    owner = _register("owner")
    outsider = _register("outsider")
    target_id, target_paths, _, run_id = _make_project_artifacts(owner, "Remove")
    survivor_id, survivor_paths, _, _ = _make_project_artifacts(owner, "Keep")

    assert client.delete(f"/projects/{target_id}", headers=outsider).status_code == 403
    with SessionLocal() as db:
        run = db.get(TrainingRun, run_id)
        run.status = "RUNNING"
        db.commit()
    active = client.delete(f"/projects/{target_id}", headers=owner)
    assert active.status_code == 409, active.text
    assert all(path.exists() for path in target_paths)

    with SessionLocal() as db:
        run = db.get(TrainingRun, run_id)
        run.status = "COMPLETED"
        db.add(AutoLabelJob(project_id=target_id, base_model="yolo26n-pose.pt", status="RUNNING"))
        db.commit()
    assert client.delete(f"/projects/{target_id}", headers=owner).status_code == 409
    with SessionLocal() as db:
        db.query(AutoLabelJob).filter_by(project_id=target_id).update({"status": "COMPLETED"})
        db.commit()

    deleted = client.delete(f"/projects/{target_id}", headers=owner)
    assert deleted.status_code == 204, deleted.text
    assert not any(path.exists() for path in target_paths)
    assert all(path.exists() for path in survivor_paths)
    assert client.get(f"/projects/{target_id}", headers=owner).status_code == 404
    assert client.get(f"/projects/{survivor_id}", headers=owner).status_code == 200
    with SessionLocal() as db:
        assert db.query(DatasetVersion).filter_by(project_id=target_id).count() == 0
        assert db.query(TrainingRun).filter_by(project_id=target_id).count() == 0
        assert db.query(Model).filter_by(project_id=target_id).count() == 0


def test_user_delete_also_cleans_owned_project_files():
    owner = _register("deleted-user")
    project_id, paths, _, _ = _make_project_artifacts(owner, "User-owned")
    with SessionLocal() as db:
        user_id = db.get(Project, project_id).owner_id
    with SessionLocal() as db:
        UserService(db).delete_user(user_id)
    assert not any(path.exists() for path in paths)
