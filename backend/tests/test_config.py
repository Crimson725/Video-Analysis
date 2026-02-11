"""Tests for app.config settings parsing."""

from app.config import Settings


class TestSettings:
    def test_cleanup_default_enabled(self, monkeypatch):
        monkeypatch.delenv("CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT", raising=False)
        settings = Settings.from_env()
        assert settings.cleanup_local_video_after_upload_default is True

    def test_cleanup_default_parses_false(self, monkeypatch):
        monkeypatch.setenv("CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT", "false")
        settings = Settings.from_env()
        assert settings.cleanup_local_video_after_upload_default is False
