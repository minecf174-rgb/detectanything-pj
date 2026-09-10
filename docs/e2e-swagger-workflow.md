# E2E Swagger Workflow

Run the API:

```bash
source venv/bin/activate
alembic upgrade head
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 1. Core Data

Create or reuse a user:

```text
GET /users
POST /users
```

Create project:

```json
POST /projects
{
  "owner_id": 3,
  "name": "Detection E2E",
  "task_type": "OBJECT_DETECTION"
}
```

Create class:

```json
POST /classes
{
  "project_id": 3,
  "name": "person",
  "class_index": 0
}
```

Upload image:

```text
POST /images/upload?project_id=3
```

## 2. Manual Annotation

Create human annotation:

```json
POST /annotations/images/{image_id}/annotation-sets
{
  "source_type": "HUMAN",
  "source_model_id": null,
  "parent_annotation_set_id": null,
  "status": "APPROVED",
  "created_by": 3,
  "annotations": [
    {
      "class_id": 3,
      "x_min": 0.1,
      "y_min": 0.1,
      "x_max": 0.6,
      "y_max": 0.7,
      "confidence": null
    }
  ]
}
```

Reload annotations:

```text
GET /annotations/images/{image_id}/annotation-sets
```

## 3. Dataset Version

Create snapshot:

```json
POST /datasets/versions
{
  "project_id": 3,
  "preprocess_config": {
    "resize": {
      "width": 640,
      "height": 640
    }
  },
  "augmentation_config": {
    "horizontal_flip": true
  },
  "created_by": 3,
  "default_split": "TRAIN",
  "auto_split": true
}
```

Build YOLO dataset:

```text
POST /datasets/versions/{version_id}/build-yolo
```

Export YOLO dataset zip:

```text
POST /datasets/versions/{version_id}/export-yolo
GET  /datasets/versions/{version_id}/export-yolo/download
```

## 4. Training

Create training run:

```json
POST /training-runs
{
  "project_id": 3,
  "version_id": 3,
  "model_type": "YOLO26_DETECT",
  "base_model": "yolo26s.pt",
  "training_config": {
    "epochs": 50,
    "imgsz": 640,
    "batch": 4,
    "patience": 25,
    "optimizer": "auto",
    "lr0": 0.01,
    "workers": 0,
    "device": "mps",
    "cache": false,
    "amp": true,
    "cos_lr": false,
    "val": true,
    "plots": true,
    "verbose": true
  }
}
```

`base_model` accepts `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`,
`yolo26l.pt`, or `yolo26x.pt`. To fine-tune a model already produced by the
same project, use `model:<model_id>`, for example `"base_model": "model:2"`.

Start real YOLO training:

```text
POST /training-runs/{training_run_id}/start-local-yolo
```

Inspect model:

```text
GET /models/project/{project_id}
GET /models/{model_id}
GET /models/{model_id}/download
```

## 5. Auto Label

Create auto-label job:

```json
POST /auto-label-jobs
{
  "project_id": 3,
  "model_id": 4,
  "image_ids": [3],
  "confidence_threshold": 0.001,
  "max_predictions_per_image": 3
}
```

Start real YOLO auto-label:

```text
POST /auto-label-jobs/{job_id}/start-local-yolo
```

Review suggestions:

```text
POST /annotations/sets/{annotation_set_id}/accept
POST /annotations/sets/{annotation_set_id}/reject
POST /annotations/sets/{annotation_set_id}/correct
```

After review, create a new dataset version and train again.
