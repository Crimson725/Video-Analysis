"""Tests for app.jobs — in-memory job state management."""

import uuid

from app.jobs import complete_job, create_job, fail_job, get_job, set_job_stage, update_job_metadata


class TestCreateJob:
    def test_returns_uuid_string(self):
        job_id = create_job()
        # Should be a valid UUID-4 string
        uuid.UUID(job_id, version=4)

    def test_new_job_has_processing_status(self):
        job_id = create_job()
        job = get_job(job_id)
        assert job is not None
        assert job["status"] == "processing"
        assert job["stage"] == "processing"


class TestGetJob:
    def test_existing_job(self):
        job_id = create_job()
        job = get_job(job_id)
        assert job is not None
        assert "status" in job

    def test_nonexistent_job_returns_none(self):
        assert get_job("nonexistent") is None


class TestCompleteJob:
    def test_sets_completed_status_and_result(self):
        job_id = create_job()
        complete_job(job_id, {"data": "result"})
        job = get_job(job_id)
        assert job is not None
        assert job["status"] == "completed"
        assert job["result"] == {"data": "result"}


class TestFailJob:
    def test_sets_failed_status_and_error(self):
        job_id = create_job()
        fail_job(job_id, "something broke")
        job = get_job(job_id)
        assert job is not None
        assert job["status"] == "failed"
        assert job["error"] == "something broke"


class TestInternalStage:
    def test_set_job_stage_updates_processing_job(self):
        job_id = create_job()
        set_job_stage(job_id, "waiting_scene_ai")
        job = get_job(job_id)
        assert job is not None
        assert job["status"] == "processing"
        assert job["stage"] == "waiting_scene_ai"

    def test_update_job_metadata_merges_fields(self):
        job_id = create_job()
        update_job_metadata(job_id, {"scene_task_id": "task-1"})
        job = get_job(job_id)
        assert job is not None
        assert job["scene_task_id"] == "task-1"
