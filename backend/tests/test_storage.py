"""Tests for app.storage (R2 media store and key helpers)."""

from unittest.mock import MagicMock

import pytest

from app.storage import (
    MediaStoreConfigError,
    MediaStoreError,
    R2MediaStore,
    build_analysis_key,
    build_frame_key,
    build_r2_endpoint,
    build_scene_key,
    build_source_video_key,
    build_summary_key,
)


class TestKeyBuilders:
    def test_build_source_video_key(self):
        assert build_source_video_key("job-123") == "jobs/job-123/input/source.mp4"

    def test_build_source_video_key_with_custom_extension(self):
        assert build_source_video_key("job-123", "mov") == "jobs/job-123/input/source.mov"

    def test_build_source_video_key_normalizes_extension(self):
        assert build_source_video_key("job-123", ".MP4") == "jobs/job-123/input/source.mp4"

    def test_build_frame_key(self):
        assert build_frame_key("job-123", "seg", 42) == "jobs/job-123/frames/seg/frame_42.jpg"

    def test_build_analysis_json_key(self):
        assert build_analysis_key("job-123", "json", 42) == "jobs/job-123/analysis/json/frame_42.json"

    def test_build_analysis_toon_key(self):
        assert build_analysis_key("job-123", "toon", 42) == "jobs/job-123/analysis/toon/frame_42.toon"

    def test_build_r2_endpoint(self):
        assert build_r2_endpoint("abc") == "https://abc.r2.cloudflarestorage.com"

    def test_build_scene_packet_key(self):
        assert build_scene_key("job-123", "packet", 2) == "jobs/job-123/scene/packets/scene_2.toon"

    def test_build_scene_narrative_key(self):
        assert (
            build_scene_key("job-123", "narrative", 2)
            == "jobs/job-123/scene/narratives/scene_2.json"
        )

    def test_build_synopsis_key(self):
        assert build_summary_key("job-123", "synopsis") == "jobs/job-123/summary/synopsis.json"


class TestR2MediaStore:
    def _make_store(self, s3_client: MagicMock | None = None) -> R2MediaStore:
        return R2MediaStore(
            account_id="account-1",
            bucket="bucket-1",
            access_key_id="key-1",
            secret_access_key="secret-1",
            default_url_ttl_seconds=1200,
            s3_client=s3_client or MagicMock(),
        )

    def test_missing_config_raises(self):
        with pytest.raises(MediaStoreConfigError):
            R2MediaStore(
                account_id="",
                bucket="bucket",
                access_key_id="key",
                secret_access_key="secret",
            )

    def test_upload_source_video_sets_content_type(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        object_key = store.upload_source_video("job-7", "/tmp/video.mp4", "video/mp4")

        assert object_key == "jobs/job-7/input/source.mp4"
        mock_client.upload_file.assert_called_once_with(
            "/tmp/video.mp4",
            "bucket-1",
            "jobs/job-7/input/source.mp4",
            ExtraArgs={"ContentType": "video/mp4"},
        )

    def test_upload_source_video_uses_extension_from_input(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        object_key = store.upload_source_video("job-7", "/tmp/video.mov", "video/quicktime", source_extension="mov")

        assert object_key == "jobs/job-7/input/source.mov"
        mock_client.upload_file.assert_called_once_with(
            "/tmp/video.mov",
            "bucket-1",
            "jobs/job-7/input/source.mov",
            ExtraArgs={"ContentType": "video/quicktime"},
        )

    def test_upload_frame_image_sets_jpeg_content_type(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        object_key = store.upload_frame_image("job-9", "det", 3, b"jpeg-bytes")

        assert object_key == "jobs/job-9/frames/det/frame_3.jpg"
        mock_client.put_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-9/frames/det/frame_3.jpg",
            Body=b"jpeg-bytes",
            ContentType="image/jpeg",
        )

    def test_upload_analysis_json_sets_json_content_type(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        object_key = store.upload_analysis_artifact("job-9", "json", 3, b'{"frame_id": 3}')

        assert object_key == "jobs/job-9/analysis/json/frame_3.json"
        mock_client.put_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-9/analysis/json/frame_3.json",
            Body=b'{"frame_id": 3}',
            ContentType="application/json",
        )

    def test_upload_analysis_toon_sets_toon_content_type(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        object_key = store.upload_analysis_artifact("job-9", "toon", 3, b"TOON_PAYLOAD")

        assert object_key == "jobs/job-9/analysis/toon/frame_3.toon"
        mock_client.put_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-9/analysis/toon/frame_3.toon",
            Body=b"TOON_PAYLOAD",
            ContentType="application/x-toon",
        )

    def test_sign_read_url_uses_ttl(self):
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = "https://signed.example/file.jpg"
        store = self._make_store(mock_client)

        signed = store.sign_read_url("jobs/job-1/frames/original/frame_0.jpg", expires_in=600)

        assert signed == "https://signed.example/file.jpg"
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "bucket-1", "Key": "jobs/job-1/frames/original/frame_0.jpg"},
            ExpiresIn=600,
        )

    def test_upload_scene_packet_sets_toon_content_type(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        object_key = store.upload_scene_artifact("job-9", "packet", 3, b"TOON_PAYLOAD")

        assert object_key == "jobs/job-9/scene/packets/scene_3.toon"
        mock_client.put_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-9/scene/packets/scene_3.toon",
            Body=b"TOON_PAYLOAD",
            ContentType="application/x-toon",
        )

    def test_upload_scene_narrative_sets_json_content_type(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        object_key = store.upload_scene_artifact("job-9", "narrative", 3, b'{"narrative":"ok"}')

        assert object_key == "jobs/job-9/scene/narratives/scene_3.json"
        mock_client.put_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-9/scene/narratives/scene_3.json",
            Body=b'{"narrative":"ok"}',
            ContentType="application/json",
        )

    def test_upload_summary_sets_json_content_type(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        object_key = store.upload_summary_artifact("job-9", "synopsis", b'{"synopsis":"ok"}')

        assert object_key == "jobs/job-9/summary/synopsis.json"
        mock_client.put_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-9/summary/synopsis.json",
            Body=b'{"synopsis":"ok"}',
            ContentType="application/json",
        )

    def test_upload_frame_wraps_sdk_errors(self):
        mock_client = MagicMock()
        mock_client.put_object.side_effect = RuntimeError("sdk error")
        store = self._make_store(mock_client)

        with pytest.raises(MediaStoreError):
            store.upload_frame_image("job-1", "face", 1, b"frame-bytes")

    def test_read_object_returns_bytes(self):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"abc123"
        mock_client.get_object.return_value = {"Body": mock_body}
        store = self._make_store(mock_client)

        payload = store.read_object("jobs/job-1/analysis/json/frame_0.json")

        assert payload == b"abc123"
        mock_client.get_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-1/analysis/json/frame_0.json",
        )

    def test_delete_object_invokes_client(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        store.delete_object("jobs/job-1/analysis/toon/frame_0.toon")

        mock_client.delete_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-1/analysis/toon/frame_0.toon",
        )

    def test_verify_object_true_when_head_succeeds(self):
        mock_client = MagicMock()
        store = self._make_store(mock_client)

        assert store.verify_object("jobs/job-1/frames/original/frame_0.jpg") is True
        mock_client.head_object.assert_called_once_with(
            Bucket="bucket-1",
            Key="jobs/job-1/frames/original/frame_0.jpg",
        )

    def test_verify_object_false_when_head_fails(self):
        mock_client = MagicMock()
        mock_client.head_object.side_effect = RuntimeError("missing")
        store = self._make_store(mock_client)

        assert store.verify_object("jobs/job-1/frames/original/frame_0.jpg") is False
