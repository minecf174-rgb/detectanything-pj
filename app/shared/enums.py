from enum import StrEnum


class AnnotationSourceType(StrEnum):
    HUMAN = "HUMAN"
    MODEL = "MODEL"
    HUMAN_CORRECTED = "HUMAN_CORRECTED"


class ReviewStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DatasetSplit(StrEnum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"

