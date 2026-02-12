"""Tests for storage-related failure paths in app.main."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from app import jobs
from app.cleanup import RETAIN_SOURCE_MARKER
from app.storage import MediaStoreConfigError, MediaStoreError


class TestProcessVideoStorageFailures:
    @patch("app.main.get_media_store", side_effect=MediaStoreConfigError("Missing R2_ACCOUNT_ID"))
    def test_missing_storage_config_marks_job_failed(self, _mock_store):
        job_id = jobs.create_job()

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "video/mp4")
        job = jobs.get_job(job_id)
        assert job is not None
        assert job["status"] == "failed"
        assert "Missing R2_ACCOUNT_ID" in job["error"]

    @patch("app.main.get_media_store")
    def test_upload_failure_marks_job_failed(self, mock_store_factory):
        mock_store = MagicMock()
        mock_store.upload_source_video.side_effect = MediaStoreError("upload failed")
        mock_store_factory.return_value = mock_store

        job_id = jobs.create_job()

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "video/mp4")
        job = jobs.get_job(job_id)
        assert job is not None
        assert job["status"] == "failed"
        assert "upload failed" in job["error"]

    @patch("app.main.scene.extract_keyframes", return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}])
    @patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)])
    @patch("app.main.scene.save_original_frames")
    @patch("app.main.analysis.analyze_frame", side_effect=RuntimeError("TOON conversion failed"))
    @patch("app.main.ModelLoader")
    @patch("app.main.get_media_store")
    def test_analysis_artifact_failure_marks_job_failed(
        self,
        mock_store_factory,
        mock_model_loader,
        _mock_analyze_frame,
        _mock_save_original,
        _mock_detect_scenes,
        _mock_extract_keyframes,
    ):
        mock_store = MagicMock()
        mock_store_factory.return_value = mock_store
        mock_model_loader.get.return_value = MagicMock()

        job_id = jobs.create_job()

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "video/mp4")
        job = jobs.get_job(job_id)
        assert job is not None
        assert job["status"] == "failed"
        assert "TOON conversion failed" in job["error"]

    @patch("app.main.scene.extract_keyframes", return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}])
    @patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)])
    @patch("app.main.scene.save_original_frames")
    @patch(
        "app.main.analysis.analyze_frame",
        return_value={
            "frame_id": 0,
            "timestamp": "00:00:01.000",
            "files": {
                "original": "jobs/job-x/frames/original/frame_0.jpg",
                "segmentation": "jobs/job-x/frames/seg/frame_0.jpg",
                "detection": "jobs/job-x/frames/det/frame_0.jpg",
                "face": "jobs/job-x/frames/face/frame_0.jpg",
            },
            "analysis": {
                "semantic_segmentation": [],
                "object_detection": [],
                "face_recognition": [],
            },
            "analysis_artifacts": {
                "json": "jobs/job-x/analysis/json/frame_0.json",
                "toon": "jobs/job-x/analysis/toon/frame_0.toon",
            },
            "metadata": {
                "provenance": {
                    "job_id": "job-x",
                    "scene_id": None,
                    "frame_id": 0,
                    "timestamp": "00:00:01.000",
                    "source_artifact_key": "jobs/job-x/frames/original/frame_0.jpg",
                },
                "model_provenance": [],
                "evidence_anchors": [],
            },
        },
    )
    @patch("app.main.ModelLoader")
    @patch("app.main.get_media_store")
    def test_verification_failure_blocks_completion(
        self,
        mock_store_factory,
        mock_model_loader,
        _mock_analyze_frame,
        _mock_save_original,
        _mock_detect_scenes,
        _mock_extract_keyframes,
    ):
        mock_store = MagicMock()
        mock_store.upload_source_video.return_value = "jobs/job-x/input/source.mp4"
        mock_store.verify_object.side_effect = (
            lambda key: key != "jobs/job-x/frames/det/frame_0.jpg"
        )
        mock_store_factory.return_value = mock_store
        mock_model_loader.get.return_value = MagicMock()

        job_id = jobs.create_job()

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")
        job = jobs.get_job(job_id)
        assert job is not None
        assert job["status"] == "failed"
        assert "Upload verification failed" in job["error"]

    @patch("app.main.get_media_store")
    def test_cleanup_policy_false_keeps_local_source(self, mock_store_factory):
        with tempfile.TemporaryDirectory() as tmp_dir:
            job_id = jobs.create_job()
            base = Path(tmp_dir) / job_id / "input"
            base.mkdir(parents=True)
            video_path = base / "source.mp4"
            video_path.write_bytes(b"video")

            mock_store = MagicMock()
            mock_store.upload_source_video.return_value = "jobs/job-keep/input/source.mp4"
            mock_store.verify_object.return_value = True
            mock_store_factory.return_value = mock_store

            with (
                patch("app.main.ModelLoader") as mock_model_loader,
                patch("app.main.scene.detect_scenes", return_value=[]),
                patch("app.main.TEMP_MEDIA_DIR", Path(tmp_dir)),
            ):
                mock_model_loader.get.return_value = MagicMock()
                from app.main import process_video

                process_video(
                    job_id,
                    str(video_path),
                    "mp4",
                    "video/mp4",
                    cleanup_local_video_after_upload=False,
                )

            assert video_path.exists()
            marker = Path(tmp_dir) / job_id / RETAIN_SOURCE_MARKER
            assert marker.exists()
