"""Tests for storage-related failure paths in app.main."""

from unittest.mock import MagicMock, patch

from app import jobs
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
