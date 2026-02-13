"""In-memory job state management."""

import uuid
from typing import Any


jobs: dict[str, dict[str, Any]] = {}


def create_job(metadata: dict[str, Any] | None = None, job_id: str | None = None) -> str:
    """Create a new job, return job_id."""
    assigned_job_id = job_id or str(uuid.uuid4())
    job_record: dict[str, Any] = {
        "status": "processing",
        "stage": "processing",
    }
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
    job_record["stage"] = "completed"
    job_record["result"] = result
    job_record.pop("error", None)
    jobs[job_id] = job_record


def fail_job(job_id: str, error: str) -> None:
    """Mark job as failed with error message."""
    job_record = jobs.get(job_id, {})
    job_record["status"] = "failed"
    job_record["stage"] = "failed"
    job_record["error"] = error
    jobs[job_id] = job_record


def set_job_stage(job_id: str, stage: str) -> None:
    """Set internal stage while preserving externally visible status semantics."""
    job_record = jobs.get(job_id)
    if job_record is None:
        return
    if job_record.get("status") == "processing":
        job_record["stage"] = stage
    jobs[job_id] = job_record


def update_job_metadata(job_id: str, metadata: dict[str, Any]) -> None:
    """Merge metadata into an existing job record."""
    job_record = jobs.get(job_id)
    if job_record is None:
        return
    job_record.update(metadata)
    jobs[job_id] = job_record
