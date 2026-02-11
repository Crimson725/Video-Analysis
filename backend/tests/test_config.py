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

    def test_scene_pipeline_defaults(self, monkeypatch):
        monkeypatch.delenv("ENABLE_SCENE_UNDERSTANDING_PIPELINE", raising=False)
        monkeypatch.delenv("SCENE_MODEL_ID", raising=False)
        monkeypatch.delenv("SYNOPSIS_MODEL_ID", raising=False)
        settings = Settings.from_env()
        assert settings.enable_scene_understanding_pipeline is False
        assert settings.scene_model_id == "gemini-2.5-flash-lite"
        assert settings.synopsis_model_id == "gemini-2.5-flash-lite"

    def test_missing_llm_fields_only_when_pipeline_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SCENE_UNDERSTANDING_PIPELINE", "true")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        settings = Settings.from_env()
        assert settings.missing_llm_fields() == ["GOOGLE_API_KEY"]

        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        settings = Settings.from_env()
        assert settings.missing_llm_fields() == []
