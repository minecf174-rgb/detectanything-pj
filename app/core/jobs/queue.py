from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JobRequest:
    job_type: str
    payload: dict[str, Any]


class JobQueue(ABC):
    @abstractmethod
    def enqueue(self, job: JobRequest) -> str:
        raise NotImplementedError


class InlineDevelopmentQueue(JobQueue):
    def enqueue(self, job: JobRequest) -> str:
        return f"dev-{job.job_type}"

