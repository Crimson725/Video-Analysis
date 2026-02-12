"""Live integration test for output-content correctness on canonical test video."""

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
def video_copy(test_video_path: str) -> str:
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


def _anchor_matches(text: str, anchors: list[str]) -> list[str]:
    """Return topic anchors that are present in normalized generated text."""
    normalized = " ".join(text.lower().split())
    return sorted({anchor for anchor in anchors if anchor in normalized})


def test_process_video_output_content_contract_for_canonical_video(
    video_copy: str,
    test_video_path: str,
    tmp_path: Path,
    gemini_probe,
    synopsis_e2e_settings,
    canonical_output_content_expectations,
):
    """Run process_video and validate contract + rubric-style output content checks."""
    del gemini_probe  # prerequisite validation via fixture side-effects
    expectations = canonical_output_content_expectations
    assert Path(test_video_path).name == expectations["video_filename"]

    job_id = jobs.create_job()
    tmp_media = Path(tmp_path) / "tmp_media"
    tmp_media.mkdir(parents=True, exist_ok=True)

    detected_scenes = detect_scenes(video_copy)
    assert len(detected_scenes) > 0
    selected_scenes = detected_scenes[:3] if len(detected_scenes) > 3 else detected_scenes

    with (
        patch("app.main.SETTINGS", synopsis_e2e_settings),
        patch("app.main.TEMP_MEDIA_DIR", tmp_media),
        patch("app.main._media_store", None),
        patch("app.main.scene.detect_scenes", return_value=selected_scenes),
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
    assert len(result.frames) > 0, "Output-content suite requires non-empty frame results"
    assert len(result.scene_narratives) > 0, "Expected scene narratives for content validation"
    assert result.video_synopsis is not None, "Expected video synopsis for content validation"

    min_narrative_chars = expectations["min_scene_narrative_chars"]
    min_key_moment_chars = expectations["min_key_moment_chars"]
    for scene_narrative in result.scene_narratives:
        assert len(scene_narrative.narrative_paragraph.strip()) >= min_narrative_chars
        assert len(scene_narrative.key_moments) > 0
        assert all(
            len(moment.strip()) >= min_key_moment_chars for moment in scene_narrative.key_moments
        )
        assert scene_narrative.artifacts.packet == (
            f"jobs/{job_id}/scene/packets/scene_{scene_narrative.scene_id}.toon"
        )
        assert scene_narrative.artifacts.narrative == (
            f"jobs/{job_id}/scene/narratives/scene_{scene_narrative.scene_id}.json"
        )
        assert scene_narrative.end_sec > scene_narrative.start_sec

    synopsis = result.video_synopsis
    assert len(synopsis.synopsis.strip()) >= expectations["min_synopsis_chars"]
    assert synopsis.artifact == f"jobs/{job_id}/summary/synopsis.json"
    assert synopsis.model == synopsis_e2e_settings.synopsis_model_id

    combined_text = " ".join(
        [synopsis.synopsis]
        + [scene.narrative_paragraph for scene in result.scene_narratives]
        + [" ".join(scene.key_moments) for scene in result.scene_narratives]
    )
    matched_anchors = _anchor_matches(combined_text, expectations["topic_anchors"])
    assert len(matched_anchors) >= expectations["min_topic_anchor_matches"], (
        "Rubric-based topical coverage check failed. "
        f"Expected at least {expectations['min_topic_anchor_matches']} anchor(s) from "
        f"{expectations['topic_anchors']}, got matches={matched_anchors}."
    )
