"""In-memory job state management."""

import uuid
from typing import Any


jobs: dict[str, dict[str, Any]] = {}


def create_job() -> str:
    """Create a new job, return job_id."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return job dict or None if not found."""
    return jobs.get(job_id)


def complete_job(job_id: str, result: dict[str, Any]) -> None:
    """Mark job as completed with result payload."""
    jobs[job_id] = {"status": "completed", "result": result}


def fail_job(job_id: str, error: str) -> None:
    """Mark job as failed with error message."""
    jobs[job_id] = {"status": "failed", "error": error}
