"""Tests for app.config settings parsing."""

from pathlib import Path

from app.config import Settings


class TestSettings:
    def test_cleanup_default_enabled(self, monkeypatch):
        monkeypatch.delenv("CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT", raising=False)
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.cleanup_local_video_after_upload_default is True

    def test_cleanup_default_parses_false(self, monkeypatch):
        monkeypatch.setenv("CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT", "false")
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.cleanup_local_video_after_upload_default is False

    def test_scene_pipeline_defaults(self, monkeypatch):
        monkeypatch.delenv("ENABLE_SCENE_UNDERSTANDING_PIPELINE", raising=False)
        monkeypatch.delenv("SCENE_AI_EXECUTION_MODE", raising=False)
        monkeypatch.delenv("ENABLE_CORPUS_PIPELINE", raising=False)
        monkeypatch.delenv("ENABLE_CORPUS_INGEST", raising=False)
        monkeypatch.delenv("SCENE_MODEL_ID", raising=False)
        monkeypatch.delenv("SYNOPSIS_MODEL_ID", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.enable_scene_understanding_pipeline is True
        assert settings.scene_ai_execution_mode == "in_process"
        assert settings.enable_corpus_pipeline is True
        assert settings.enable_corpus_ingest is True
        assert settings.scene_model_id == "gemini-3-flash-preview"
        assert settings.synopsis_model_id == "gemini-3-flash-preview"
        assert settings.embedding_model_id == "gemini-embedding-001"

    def test_scene_ai_queue_settings_parse(self, monkeypatch):
        monkeypatch.setenv("SCENE_AI_EXECUTION_MODE", "queue")
        monkeypatch.setenv("SCENE_AI_MAX_ATTEMPTS", "7")
        monkeypatch.setenv("SCENE_AI_RETRY_BACKOFF_SECONDS", "4")
        monkeypatch.setenv("SCENE_AI_RETRY_BACKOFF_MULTIPLIER", "3")
        monkeypatch.setenv("SCENE_AI_RETRY_BACKOFF_MAX_SECONDS", "90")
        monkeypatch.setenv("SCENE_AI_LEASE_TIMEOUT_SECONDS", "45")
        monkeypatch.setenv("SCENE_AI_FAILURE_POLICY", "fallback_empty")
        monkeypatch.setenv("SCENE_AI_WORKER_POLL_INTERVAL_SECONDS", "9")
        monkeypatch.setenv("SCENE_AI_PROMPT_VERSION", "v2")
        monkeypatch.setenv("SCENE_AI_RUNTIME_VERSION", "rt-2026")
        settings = Settings.from_env(autoload_dotenv=False)

        assert settings.scene_ai_execution_mode == "queue"
        assert settings.scene_ai_max_attempts == 7
        assert settings.scene_ai_retry_backoff_seconds == 4
        assert settings.scene_ai_retry_backoff_multiplier == 3
        assert settings.scene_ai_retry_backoff_max_seconds == 90
        assert settings.scene_ai_lease_timeout_seconds == 45
        assert settings.scene_ai_failure_policy == "fallback_empty"
        assert settings.scene_ai_worker_poll_interval_seconds == 9
        assert settings.scene_ai_prompt_version == "v2"
        assert settings.scene_ai_runtime_version == "rt-2026"

    def test_corpus_settings_parse(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CORPUS_PIPELINE", "true")
        monkeypatch.setenv("ENABLE_CORPUS_INGEST", "true")
        monkeypatch.setenv("GRAPH_BACKEND", "memory")
        monkeypatch.setenv("VECTOR_BACKEND", "memory")
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.enable_corpus_pipeline is True
        assert settings.enable_corpus_ingest is True
        assert settings.graph_backend == "memory"
        assert settings.vector_backend == "memory"

    def test_missing_llm_fields_only_when_pipeline_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SCENE_UNDERSTANDING_PIPELINE", "true")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.missing_llm_fields() == ["GOOGLE_API_KEY"]

        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.missing_llm_fields() == []

    def test_missing_embedding_fields_for_gemini_model(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("EMBEDDING_MODEL_ID", "gemini-embedding-001")
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.missing_embedding_fields() == ["GOOGLE_API_KEY"]

        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.missing_embedding_fields() == []

    def test_autoloads_google_api_key_from_dotenv_local(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        env_local = tmp_path / ".env.local"
        env_local.write_text("GOOGLE_API_KEY=file-key\n", encoding="utf-8")

        settings = Settings.from_env(dotenv_files=(env_local,))

        assert settings.google_api_key == "file-key"

    def test_environment_value_overrides_dotenv(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        env_local = tmp_path / ".env.local"
        env_local.write_text("GOOGLE_API_KEY=file-key\n", encoding="utf-8")

        settings = Settings.from_env(dotenv_files=(env_local,))

        assert settings.google_api_key == "env-key"
