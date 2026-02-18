"""Infrastructure adapter for in-memory job state storage."""

from __future__ import annotations

from typing import Any

from app import jobs


class InMemoryJobStore:
    """Adapter boundary over app.jobs module."""

    def create(
        self,
        *,
        job_id: str,
        metadata: dict[str, Any] | None = None,
        status: str = "queued",
    ) -> str:
        stage = "queued" if status == "queued" else "processing"
        return jobs.create_job(
            metadata=metadata,
            job_id=job_id,
            status=status,
            stage=stage,
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        return jobs.get_job(job_id)

    def set_stage(self, job_id: str, stage: str) -> None:
        jobs.set_job_stage(job_id, stage)

    def set_processing(self, job_id: str, current_stage: str = "processing") -> None:
        jobs.set_job_status(job_id, "processing")
        jobs.set_job_stage(job_id, current_stage)

    def complete(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        pipeline: dict[str, Any] | None = None,
    ) -> None:
        jobs.complete_job(job_id, result, pipeline=pipeline)

    def fail(
        self,
        job_id: str,
        error: str,
        *,
        failed_stage: str | None = None,
        pipeline: dict[str, Any] | None = None,
    ) -> None:
        jobs.fail_job(
            job_id,
            error,
            failed_stage=failed_stage,
            pipeline=pipeline,
        )

    def update_metadata(self, job_id: str, metadata: dict[str, Any]) -> None:
        jobs.update_job_metadata(job_id, metadata)
