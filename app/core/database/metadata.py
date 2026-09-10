from app.core.database.session import Base

# Import ORM models so Alembic can discover the full metadata graph.
from app.modules.annotations.models import Annotation, AnnotationSet  # noqa: F401
from app.modules.auth.models import AuthSession  # noqa: F401
from app.modules.auto_label.models import AutoLabelJob, AutoLabelJobItem  # noqa: F401
from app.modules.classes.models import LabelClass  # noqa: F401
from app.modules.datasets.models import DatasetVersion, DatasetVersionItem  # noqa: F401
from app.modules.images.models import Image  # noqa: F401
from app.modules.models.models import Model, ModelArtifact  # noqa: F401
from app.modules.projects.models import Project, User  # noqa: F401
from app.modules.training.models import TrainingMetric, TrainingRun  # noqa: F401

target_metadata = Base.metadata
