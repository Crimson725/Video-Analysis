"""Tests for CV-only processing flow in app.main.process_video."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import jobs


def _frame_payload(job_id: str) -> dict:
    return {
        "frame_id": 0,
        "timestamp": "00:00:01.000",
        "files": {
            "original": f"jobs/{job_id}/frames/original/frame_0.jpg",
            "segmentation": f"jobs/{job_id}/frames/seg/frame_0.jpg",
            "detection": f"jobs/{job_id}/frames/det/frame_0.jpg",
            "face": f"jobs/{job_id}/frames/face/frame_0.jpg",
        },
        "analysis": {
            "semantic_segmentation": [],
            "object_detection": [],
            "face_recognition": [
                {
                    "face_id": 1,
                    "identity_id": "face_1",
                    "confidence": 0.95,
                    "coordinates": [1, 2, 20, 30],
                }
            ],
            "enrichment": {},
        },
        "analysis_artifacts": {
            "json": f"jobs/{job_id}/analysis/json/frame_0.json",
        },
        "metadata": {
            "provenance": {
                "job_id": job_id,
                "scene_id": None,
                "frame_id": 0,
                "timestamp": "00:00:01.000",
                "source_artifact_key": f"jobs/{job_id}/frames/original/frame_0.jpg",
            },
            "model_provenance": [],
            "evidence_anchors": [],
        },
    }


def test_process_video_skips_post_cv_pipelines_even_when_enabled():
    job_id = jobs.create_job()

    with (
        patch("app.main.get_media_store") as mock_store_factory,
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[
                {"frame_id": 0, "scene_id": 0, "timestamp": "00:00:00.500", "image": object()}
            ],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame", return_value=_frame_payload(job_id)),
        patch("app.main.analysis.run_object_tracking_summary") as mock_object_summary,
        patch("app.main.scene.extract_tracking_frames") as mock_extract_tracking_frames,
        patch("app.main.analysis.run_face_identity_pipeline") as mock_face_identity,
        patch("app.main.analysis.run_person_tracking_fusion") as mock_person_fusion,
        patch(
            "app.main.SETTINGS",
            SimpleNamespace(
                cleanup_local_video_after_upload_default=True,
                r2_url_ttl_seconds=3600,
                enable_face_identity_pipeline=True,
                face_identity_sample_fps=2,
                face_identity_max_samples_per_scene=10,
            ),
        ),
    ):
        mock_store = MagicMock(name="store")
        mock_store.upload_source_video.return_value = f"jobs/{job_id}/input/source.mp4"
        mock_store.verify_object.return_value = True
        mock_store_factory.return_value = mock_store
        mock_model_loader.get.return_value = MagicMock()

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert set(job["result"].keys()) == {"job_id", "pipeline", "frames"}
    mock_object_summary.assert_not_called()
    mock_extract_tracking_frames.assert_not_called()
    mock_face_identity.assert_not_called()
    mock_person_fusion.assert_not_called()


def test_process_video_result_contains_only_cv_outputs():
    job_id = jobs.create_job()

    with (
        patch("app.main.get_media_store") as mock_store_factory,
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[
                {"frame_id": 0, "scene_id": 0, "timestamp": "00:00:00.500", "image": object()}
            ],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame", return_value=_frame_payload(job_id)),
        patch(
            "app.main.SETTINGS",
            SimpleNamespace(
                cleanup_local_video_after_upload_default=True,
                r2_url_ttl_seconds=3600,
                enable_face_identity_pipeline=False,
            ),
        ),
    ):
        mock_store = MagicMock(name="store")
        mock_store.upload_source_video.return_value = f"jobs/{job_id}/input/source.mp4"
        mock_store.verify_object.return_value = True
        mock_store_factory.return_value = mock_store
        mock_model_loader.get.return_value = MagicMock()

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert set(job["result"].keys()) == {"job_id", "pipeline", "frames"}
    assert len(job["result"]["frames"]) == 1
