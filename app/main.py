from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config.settings import get_settings
from app.core.database.session import SessionLocal
from app.modules.auth.router import router as auth_router
from app.modules.annotations.router import router as annotations_router
from app.modules.auto_label.router import router as auto_label_router
from app.modules.classes.router import router as classes_router
from app.modules.datasets.router import router as datasets_router
from app.modules.images.router import router as images_router
from app.modules.models.router import router as models_router
from app.modules.projects.router import router as projects_router
from app.modules.training.router import router as training_router
from app.modules.training.service import TrainingService


@asynccontextmanager
async def lifespan(_app: FastAPI):
    with SessionLocal() as db:
        TrainingService.recover_interrupted_runs(db)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="CV Training Platform", lifespan=lifespan)
    settings = get_settings()
    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    allow_all_origins = "*" in origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all_origins else origins,
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router, prefix="/auth", tags=["auth"])
    app.include_router(projects_router, prefix="/projects", tags=["projects"])
    app.include_router(images_router, prefix="/images", tags=["images"])
    app.include_router(classes_router, prefix="/classes", tags=["classes"])
    app.include_router(annotations_router, prefix="/annotations", tags=["annotations"])
    app.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
    app.include_router(training_router, prefix="/training-runs", tags=["training"])
    app.include_router(models_router, prefix="/models", tags=["models"])
    app.include_router(auto_label_router, prefix="/auto-label-jobs", tags=["auto-label"])

    return app


app = create_app()
