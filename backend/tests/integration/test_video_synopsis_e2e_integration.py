"""Live integration test for end-to-end video synopsis generation."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app import jobs
from app.main import process_video
from app.scene import detect_scenes
from app.schemas import JobResult


pytestmark = [pytest.mark.integration, pytest.mark.external_api]


@pytest.fixture()
def video_copy(test_video_path):
    """Copy the test video to a temp file so process_video can delete it."""
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


def test_process_video_generates_live_synopsis(
    video_copy,
    tmp_path,
    gemini_probe,
    synopsis_e2e_settings,
    assert_scene_llm_smoke_result,
):
    """Run process_video with real Gemini and validate synopsis result structure."""
    del gemini_probe  # probe fixture is used for prerequisite validation.
    job_id = jobs.create_job()
    tmp_media = Path(tmp_path) / "tmp_media"
    tmp_media.mkdir(parents=True, exist_ok=True)
    detected_scenes = detect_scenes(video_copy)
    assert len(detected_scenes) > 0
    single_scene = [detected_scenes[0]]

    with (
        patch("app.main.SETTINGS", synopsis_e2e_settings),
        patch("app.main.TEMP_MEDIA_DIR", tmp_media),
        patch("app.main._media_store", None),
        patch("app.main.scene.detect_scenes", return_value=single_scene),
        patch("app.video_understanding.LANGGRAPH_AVAILABLE", False),
    ):
        process_video(job_id, video_copy, "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    if job["status"] == "failed":
        error = str(job.get("error", ""))
        if "RESOURCE_EXHAUSTED" in error or "rate-limits" in error or "quota" in error.lower():
            pytest.skip(f"Gemini quota/rate limit unavailable for this run: {error}")
    assert job["status"] == "completed", (
        f"Expected 'completed', got '{job['status']}'. "
        f"Error: {job.get('error', 'N/A')}"
    )

    result = JobResult(**job["result"])
    assert_scene_llm_smoke_result(
        result,
        job_id=job_id,
        synopsis_model_id=synopsis_e2e_settings.synopsis_model_id,
    )
