from fastapi import HTTPException

from app.modules.classes.models import LabelClass
from app.modules.models.models import Model


def model_class_mapping(model: Model, classes: list[LabelClass]) -> dict[int, LabelClass]:
    """Map a model's output index to this project's current class records."""
    if model.source_type != "IMPORTED" or not model.class_names:
        return {label_class.class_index: label_class for label_class in classes}
    by_name = {label_class.name: label_class for label_class in classes}
    missing = [name for name in model.class_names if name not in by_name]
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"Imported model classes are missing from this project: {', '.join(missing)}",
        )
    return {index: by_name[name] for index, name in enumerate(model.class_names)}
