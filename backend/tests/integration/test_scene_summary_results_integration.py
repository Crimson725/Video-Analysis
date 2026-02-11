"""Integration-style test for scene summary outputs in process_video."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import jobs
from app.main import process_video


pytestmark = pytest.mark.integration


def test_process_video_includes_scene_outputs(tmp_path):
    video_path = Path(tmp_path) / "source.mp4"
    video_path.write_bytes(b"video")

    fake_store = MagicMock()
    fake_store.upload_source_video.return_value = "jobs/job-1/input/source.mp4"
    fake_store.verify_object.return_value = True

    frame_payload = {
        "frame_id": 0,
        "timestamp": "00:00:01.000",
        "files": {
            "original": "jobs/job-1/frames/original/frame_0.jpg",
            "segmentation": "jobs/job-1/frames/seg/frame_0.jpg",
            "detection": "jobs/job-1/frames/det/frame_0.jpg",
            "face": "jobs/job-1/frames/face/frame_0.jpg",
        },
        "analysis": {
            "semantic_segmentation": [],
            "object_detection": [],
            "face_recognition": [],
        },
        "analysis_artifacts": {
            "json": "jobs/job-1/analysis/json/frame_0.json",
            "toon": "jobs/job-1/analysis/toon/frame_0.toon",
        },
    }

    scene_outputs = {
        "scene_narratives": [
            {
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "narrative_paragraph": "Scene summary.",
                "key_moments": ["moment 1"],
                "artifacts": {
                    "packet": "jobs/job-1/scene/packets/scene_0.toon",
                    "narrative": "jobs/job-1/scene/narratives/scene_0.json",
                },
            }
        ],
        "video_synopsis": {
            "synopsis": "Video synopsis.",
            "artifact": "jobs/job-1/summary/synopsis.json",
            "model": "gemini-2.5-flash-lite",
        },
    }

    settings = SimpleNamespace(
        cleanup_local_video_after_upload_default=True,
        enable_scene_understanding_pipeline=True,
    )

    job_id = jobs.create_job()
    with (
        patch("app.main.SETTINGS", settings),
        patch("app.main.TEMP_MEDIA_DIR", Path(tmp_path) / "tmp_media"),
        patch("app.main.get_media_store", return_value=fake_store),
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame", return_value=frame_payload),
        patch(
            "app.main.video_understanding.run_scene_understanding_pipeline",
            return_value=scene_outputs,
        ),
        patch("app.main.ModelLoader") as mock_model_loader,
    ):
        mock_model_loader.get.return_value = MagicMock()
        process_video(job_id, str(video_path), "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert len(job["result"]["scene_narratives"]) == 1
    assert job["result"]["video_synopsis"]["artifact"] == "jobs/job-1/summary/synopsis.json"
