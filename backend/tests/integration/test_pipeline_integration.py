"""Integration tests for the end-to-end process_video pipeline with real video."""

import shutil
import tempfile

import pytest

from app import jobs
from app.main import process_video
from app.schemas import JobResult


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture: copy test video to a temp file (process_video deletes the original)
# ---------------------------------------------------------------------------

@pytest.fixture()
def video_copy(test_video_path):
    """Copy the test video to a temp file so process_video can safely delete it."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    shutil.copy2(test_video_path, path)
    return path


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Ensure every test starts with a clean job store."""
    jobs.jobs.clear()
    yield
    jobs.jobs.clear()


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
            assert frame.analysis_artifacts.toon, "toon analysis artifact path is empty"

            # analysis: all three lists present (may be empty for some frames)
            assert isinstance(frame.analysis.semantic_segmentation, list)
            assert isinstance(frame.analysis.object_detection, list)
            assert isinstance(frame.analysis.face_recognition, list)
