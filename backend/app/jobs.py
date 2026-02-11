"""In-memory job state management."""

import uuid
from typing import Any


jobs: dict[str, dict[str, Any]] = {}


def create_job(metadata: dict[str, Any] | None = None, job_id: str | None = None) -> str:
    """Create a new job, return job_id."""
    assigned_job_id = job_id or str(uuid.uuid4())
    job_record: dict[str, Any] = {"status": "processing"}
    if metadata:
        job_record.update(metadata)
    jobs[assigned_job_id] = job_record
    return assigned_job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return job dict or None if not found."""
    return jobs.get(job_id)


def complete_job(job_id: str, result: dict[str, Any]) -> None:
    """Mark job as completed with result payload."""
    job_record = jobs.get(job_id, {})
    job_record["status"] = "completed"
    job_record["result"] = result
    job_record.pop("error", None)
    jobs[job_id] = job_record


def fail_job(job_id: str, error: str) -> None:
    """Mark job as failed with error message."""
    job_record = jobs.get(job_id, {})
    job_record["status"] = "failed"
    job_record["error"] = error
    jobs[job_id] = job_record
