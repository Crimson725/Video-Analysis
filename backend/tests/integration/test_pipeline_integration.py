"""Integration tests for the end-to-end process_video pipeline with real video."""

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile

import pytest

from app import jobs
from app.config import Settings
from app.main import process_video
from app.schemas import JobResult
from app.storage import (
    build_analysis_key,
    build_corpus_key,
    build_frame_key,
    build_scene_key,
    build_source_video_key,
    build_summary_key,
)


pytestmark = pytest.mark.integration


class InMemoryMediaStore:
    """In-memory MediaStore substitute for integration tests without R2."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_source_video(
        self,
        job_id: str,
        file_path: str,
        content_type: str,
        source_extension: str | None = None,
    ) -> str:
        del content_type
        object_key = build_source_video_key(job_id, source_extension=source_extension)
        self.objects[object_key] = Path(file_path).read_bytes()
        return object_key

    def upload_frame_image(self, job_id: str, frame_kind: str, frame_id: int, image_bytes: bytes) -> str:
        object_key = build_frame_key(job_id, frame_kind, frame_id)
        self.objects[object_key] = image_bytes
        return object_key

    def upload_analysis_artifact(self, job_id: str, artifact_kind: str, frame_id: int, payload: bytes) -> str:
        object_key = build_analysis_key(job_id, artifact_kind, frame_id)
        self.objects[object_key] = payload
        return object_key

    def upload_scene_artifact(self, job_id: str, artifact_kind: str, scene_id: int, payload: bytes) -> str:
        object_key = build_scene_key(job_id, artifact_kind, scene_id)
        self.objects[object_key] = payload
        return object_key

    def upload_summary_artifact(self, job_id: str, artifact_kind: str, payload: bytes) -> str:
        object_key = build_summary_key(job_id, artifact_kind)
        self.objects[object_key] = payload
        return object_key

    def upload_corpus_artifact(
        self,
        job_id: str,
        artifact_kind: str,
        payload: bytes,
        filename: str = "bundle.json",
    ) -> str:
        object_key = build_corpus_key(job_id, artifact_kind, filename=filename)
        self.objects[object_key] = payload
        return object_key

    def verify_object(self, object_key: str) -> bool:
        return object_key in self.objects

    def sign_read_url(self, object_key: str, expires_in: int | None = None) -> str:
        del expires_in
        return f"https://signed.example/{object_key}"


# ---------------------------------------------------------------------------
# Fixture: copy test video to a temp file (process_video deletes the original)
# ---------------------------------------------------------------------------

@pytest.fixture()
def video_copy(test_video_path):
    """Copy the test video to a temp file so process_video can safely delete it."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    shutil.copy2(test_video_path, path)
    return path


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Ensure every test starts with a clean job store."""
    jobs.jobs.clear()
    yield
    jobs.jobs.clear()


@pytest.fixture(autouse=True)
def _configure_process_video_for_local_integration(monkeypatch, tmp_path):
    """Use in-memory object storage and disable corpus/LLM side effects for this module."""
    from app import main as app_main

    media_store = InMemoryMediaStore()
    base_settings = Settings.from_env()
    test_settings = replace(
        base_settings,
        enable_scene_understanding_pipeline=False,
        enable_corpus_pipeline=False,
        enable_corpus_ingest=False,
        cleanup_local_video_after_upload_default=True,
    )
    temp_media_dir = tmp_path / "tmp_media"
    temp_media_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(app_main, "SETTINGS", test_settings)
    monkeypatch.setattr(app_main, "TEMP_MEDIA_DIR", temp_media_dir)
    monkeypatch.setattr(app_main, "_media_store", None)
    monkeypatch.setattr(app_main, "get_media_store", lambda: media_store)


# ---------------------------------------------------------------------------
# 8.2 — process_video completes successfully
# ---------------------------------------------------------------------------

class TestProcessVideoIntegration:
    def test_completes_with_status_completed(self, video_copy):
        job_id = jobs.create_job()

        process_video(job_id, video_copy)

        job = jobs.get_job(job_id)
        assert job is not None
        assert job["status"] == "completed", (
            f"Expected 'completed', got '{job['status']}'. "
            f"Error: {job.get('error', 'N/A')}"
        )


# ---------------------------------------------------------------------------
# 8.3 — Result conforms to JobResult Pydantic model
# ---------------------------------------------------------------------------

class TestProcessVideoResultSchema:
    def test_result_parseable_by_job_result_model(self, video_copy):
        job_id = jobs.create_job()
        process_video(job_id, video_copy)

        job = jobs.get_job(job_id)
        assert job["status"] == "completed"

        # This will raise ValidationError if the payload doesn't conform
        result = JobResult(**job["result"])

        assert isinstance(result.frames, list)
        assert len(result.frames) > 0, "Expected at least one frame in results"


# ---------------------------------------------------------------------------
# 8.4 — Each frame has all analysis types and file paths
# ---------------------------------------------------------------------------

class TestProcessVideoFrameContent:
    def test_each_frame_has_files_and_analysis(self, video_copy):
        job_id = jobs.create_job()
        process_video(job_id, video_copy)

        job = jobs.get_job(job_id)
        assert job["status"] == "completed"
        result = JobResult(**job["result"])

        for frame in result.frames:
            # files: all four paths are non-empty
            assert frame.files.original, "original file path is empty"
            assert frame.files.segmentation, "segmentation file path is empty"
            assert frame.files.detection, "detection file path is empty"
            assert frame.files.face, "face file path is empty"
            assert frame.analysis_artifacts.json_artifact, "json analysis artifact path is empty"

            # analysis: all three lists present (may be empty for some frames)
            assert isinstance(frame.analysis.semantic_segmentation, list)
            assert isinstance(frame.analysis.object_detection, list)
            assert isinstance(frame.analysis.face_recognition, list)
