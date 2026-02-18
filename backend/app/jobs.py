"""In-memory job state management."""

import uuid
from typing import Any


jobs: dict[str, dict[str, Any]] = {}


def create_job(
    metadata: dict[str, Any] | None = None,
    job_id: str | None = None,
    *,
    status: str = "processing",
    stage: str | None = None,
) -> str:
    """Create a new job, return job_id."""
    assigned_job_id = job_id or str(uuid.uuid4())
    initial_stage = stage or ("queued" if status == "queued" else "processing")
    job_record: dict[str, Any] = {
        "status": status,
        "stage": initial_stage,
        "current_stage": initial_stage,
        "failed_stage": None,
    }
    if metadata:
        job_record.update(metadata)
        if "stage" in metadata and "current_stage" not in metadata:
            job_record["current_stage"] = metadata["stage"]
    jobs[assigned_job_id] = job_record
    return assigned_job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return job dict or None if not found."""
    return jobs.get(job_id)


def complete_job(
    job_id: str,
    result: dict[str, Any],
    *,
    pipeline: dict[str, Any] | None = None,
) -> None:
    """Mark job as completed with result payload."""
    job_record = jobs.get(job_id, {})
    job_record["status"] = "completed"
    job_record["stage"] = "completed"
    job_record["current_stage"] = "completed"
    job_record["result"] = result
    job_record.pop("error", None)
    job_record["failed_stage"] = None
    if pipeline is not None:
        job_record["pipeline"] = pipeline
    jobs[job_id] = job_record


def fail_job(
    job_id: str,
    error: str,
    *,
    failed_stage: str | None = None,
    pipeline: dict[str, Any] | None = None,
) -> None:
    """Mark job as failed with error message."""
    job_record = jobs.get(job_id, {})
    job_record["status"] = "failed"
    job_record["stage"] = "failed"
    job_record["current_stage"] = "failed"
    job_record["error"] = error
    job_record["failed_stage"] = failed_stage
    if pipeline is not None:
        job_record["pipeline"] = pipeline
    jobs[job_id] = job_record


def set_job_stage(job_id: str, stage: str) -> None:
    """Set internal stage while preserving externally visible status semantics."""
    job_record = jobs.get(job_id)
    if job_record is None:
        return
    if job_record.get("status") in {"processing", "queued"}:
        job_record["stage"] = stage
        job_record["current_stage"] = stage
    jobs[job_id] = job_record


def set_job_status(job_id: str, status: str) -> None:
    """Set externally visible status without dropping other metadata."""
    job_record = jobs.get(job_id)
    if job_record is None:
        return
    job_record["status"] = status
    jobs[job_id] = job_record


def update_job_metadata(job_id: str, metadata: dict[str, Any]) -> None:
    """Merge metadata into an existing job record."""
    job_record = jobs.get(job_id)
    if job_record is None:
        return
    job_record.update(metadata)
    jobs[job_id] = job_record
