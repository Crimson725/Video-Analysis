"""Tests for app.config settings parsing."""

from pathlib import Path

import pytest

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
        monkeypatch.delenv("SCENE_AI_EXECUTION_MODE", raising=False)
        monkeypatch.delenv("ENABLE_CORPUS_PIPELINE", raising=False)
        monkeypatch.delenv("ENABLE_CORPUS_INGEST", raising=False)
        monkeypatch.delenv("SCENE_MODEL_ID", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.scene_ai_execution_mode == "in_process"
        assert settings.enable_corpus_pipeline is True
        assert settings.enable_corpus_ingest is True
        assert settings.scene_model_id == "gemini-3-flash-preview"
        assert settings.embedding_model_id == "gemini-embedding-001"

    def test_parallel_chunked_tracking_defaults(self, monkeypatch):
        monkeypatch.delenv("ENABLE_PARALLEL_CHUNKED_TRACKING_PIPELINE", raising=False)
        monkeypatch.delenv("ENABLE_PIPELINE_BRANCH_CONCURRENCY", raising=False)
        monkeypatch.delenv("PIPELINE_FRAME_BRANCH_WORKER_BUDGET", raising=False)
        monkeypatch.delenv("PIPELINE_CHUNK_BRANCH_WORKER_BUDGET", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_GROUND_TRUTH_BACKEND", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_OUTPUT_MODE", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_SCENE_TOP_ENTITIES", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_SCENE_GRID_WIDTH", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_SCENE_GRID_HEIGHT", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_SCENE_PATH_MAX_SEGMENTS", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_CHUNK_DURATION_SEC", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_OVERLAP_SEC", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_SAMPLE_FPS", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_CHUNK_MAX_WORKERS", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_TRACKER_CONFIG", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_BACKEND_STRATEGY", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_STITCH_STRATEGY", raising=False)
        monkeypatch.delenv("PARALLEL_TRACKING_ZONE_STRATEGY", raising=False)

        settings = Settings.from_env(autoload_dotenv=False)

        assert settings.enable_parallel_chunked_tracking_pipeline is False
        assert settings.enable_pipeline_branch_concurrency in {True, False}
        assert settings.pipeline_frame_branch_worker_budget == 1
        assert settings.pipeline_chunk_branch_worker_budget >= 1
        assert settings.parallel_tracking_ground_truth_backend == "sqlite"
        assert settings.parallel_tracking_output_mode == "summary_v2"
        assert settings.parallel_tracking_scene_top_entities == 30
        assert settings.parallel_tracking_scene_grid_width == 12
        assert settings.parallel_tracking_scene_grid_height == 7
        assert settings.parallel_tracking_scene_path_max_segments == 6
        assert settings.parallel_tracking_chunk_duration_sec == 300
        assert settings.parallel_tracking_overlap_sec == 15
        assert settings.parallel_tracking_sample_fps == 10
        assert settings.parallel_tracking_chunk_max_workers >= 1
        assert settings.parallel_tracking_tracker_config == "botsort_reid.yaml"
        assert settings.parallel_tracking_backend_strategy == "default"
        assert settings.parallel_tracking_stitch_strategy == "default"
        assert settings.parallel_tracking_zone_strategy == "grid3x3"

    def test_parallel_chunked_tracking_scene_summary_settings_parse(self, monkeypatch):
        monkeypatch.setenv("ENABLE_PIPELINE_BRANCH_CONCURRENCY", "true")
        monkeypatch.setenv("PIPELINE_FRAME_BRANCH_WORKER_BUDGET", "1")
        monkeypatch.setenv("PIPELINE_CHUNK_BRANCH_WORKER_BUDGET", "3")
        monkeypatch.setenv("PARALLEL_TRACKING_GROUND_TRUTH_BACKEND", "parquet")
        monkeypatch.setenv("PARALLEL_TRACKING_OUTPUT_MODE", "dual")
        monkeypatch.setenv("PARALLEL_TRACKING_SCENE_TOP_ENTITIES", "24")
        monkeypatch.setenv("PARALLEL_TRACKING_SCENE_GRID_WIDTH", "10")
        monkeypatch.setenv("PARALLEL_TRACKING_SCENE_GRID_HEIGHT", "6")
        monkeypatch.setenv("PARALLEL_TRACKING_SCENE_PATH_MAX_SEGMENTS", "4")
        monkeypatch.setenv("PARALLEL_TRACKING_CHUNK_MAX_WORKERS", "5")
        monkeypatch.setenv("PARALLEL_TRACKING_BACKEND_STRATEGY", "default")
        monkeypatch.setenv("PARALLEL_TRACKING_STITCH_STRATEGY", "default")
        monkeypatch.setenv("PARALLEL_TRACKING_ZONE_STRATEGY", "grid3x3")

        settings = Settings.from_env(autoload_dotenv=False)

        assert settings.enable_pipeline_branch_concurrency is True
        assert settings.pipeline_frame_branch_worker_budget == 1
        assert settings.pipeline_chunk_branch_worker_budget == 3
        assert settings.parallel_tracking_ground_truth_backend == "parquet"
        assert settings.parallel_tracking_output_mode == "dual"
        assert settings.parallel_tracking_scene_top_entities == 24
        assert settings.parallel_tracking_scene_grid_width == 10
        assert settings.parallel_tracking_scene_grid_height == 6
        assert settings.parallel_tracking_scene_path_max_segments == 4
        assert settings.parallel_tracking_chunk_max_workers == 5

    def test_scene_ai_queue_settings_parse(self, monkeypatch):
        monkeypatch.setenv("SCENE_AI_EXECUTION_MODE", "queue")
        monkeypatch.setenv("SCENE_AI_MAX_ATTEMPTS", "7")
        monkeypatch.setenv("SCENE_AI_RETRY_BACKOFF_SECONDS", "4")
        monkeypatch.setenv("SCENE_AI_RETRY_BACKOFF_MULTIPLIER", "3")
        monkeypatch.setenv("SCENE_AI_RETRY_BACKOFF_MAX_SECONDS", "90")
        monkeypatch.setenv("SCENE_AI_LEASE_TIMEOUT_SECONDS", "45")
        monkeypatch.setenv("SCENE_AI_FAILURE_POLICY", "fallback_empty")
        monkeypatch.setenv("SCENE_AI_WORKER_POLL_INTERVAL_SECONDS", "9")
        settings = Settings.from_env(autoload_dotenv=False)

        assert settings.scene_ai_execution_mode == "queue"
        assert settings.scene_ai_max_attempts == 7
        assert settings.scene_ai_retry_backoff_seconds == 4
        assert settings.scene_ai_retry_backoff_multiplier == 3
        assert settings.scene_ai_retry_backoff_max_seconds == 90
        assert settings.scene_ai_lease_timeout_seconds == 45
        assert settings.scene_ai_failure_policy == "fallback_empty"
        assert settings.scene_ai_worker_poll_interval_seconds == 9

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

    def test_face_identity_settings_parse(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FACE_IDENTITY_PIPELINE", "true")
        monkeypatch.setenv("FACE_IDENTITY_BACKEND", "mps")
        monkeypatch.setenv("FACE_IDENTITY_SAMPLE_FPS", "6")
        monkeypatch.setenv("FACE_IDENTITY_MAX_SAMPLES_PER_SCENE", "42")
        monkeypatch.setenv("FACE_IDENTITY_SCENE_SIMILARITY_THRESHOLD", "0.81")
        monkeypatch.setenv("FACE_IDENTITY_VIDEO_SIMILARITY_THRESHOLD", "0.84")
        monkeypatch.setenv("FACE_IDENTITY_AMBIGUITY_MARGIN", "0.02")
        monkeypatch.setenv("FACE_IDENTITY_EMBEDDING_DIMENSION", "256")
        monkeypatch.setenv("FACE_IDENTITY_MODEL_ID", "edgeface_xs_gamma_06")
        monkeypatch.setenv("FACE_IDENTITY_WEIGHTS_PATH", "/tmp/edgeface.pt")
        monkeypatch.setenv("FACE_IDENTITY_ARCFACE_MODEL_NAME", "buffalo_l")
        monkeypatch.setenv(
            "FACE_IDENTITY_ARCFACE_PROVIDER_ORDER",
            "CoreMLExecutionProvider,CPUExecutionProvider",
        )
        monkeypatch.setenv("FACE_IDENTITY_ARCFACE_FALLBACK_BEHAVIOR", "cpu")

        settings = Settings.from_env(autoload_dotenv=False)

        assert settings.enable_face_identity_pipeline is True
        assert settings.face_identity_backend == "mps"
        assert settings.face_identity_sample_fps == 6
        assert settings.face_identity_max_samples_per_scene == 42
        assert settings.face_identity_scene_similarity_threshold == 0.81
        assert settings.face_identity_video_similarity_threshold == 0.84
        assert settings.face_identity_ambiguity_margin == 0.02
        assert settings.face_identity_embedding_dimension == 256
        assert settings.face_identity_model_id == "edgeface_xs_gamma_06"
        assert settings.face_identity_weights_path == "/tmp/edgeface.pt"
        assert settings.face_identity_arcface_model_name == "buffalo_l"
        assert settings.face_identity_arcface_provider_order == (
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        )
        assert settings.face_identity_arcface_fallback_behavior == "cpu"

    def test_face_identity_pipeline_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ENABLE_FACE_IDENTITY_PIPELINE", raising=False)

        settings = Settings.from_env(autoload_dotenv=False)

        assert settings.enable_face_identity_pipeline is True

    def test_face_identity_defaults_use_edgeface_profile(self, monkeypatch):
        monkeypatch.delenv("FACE_IDENTITY_SCENE_SIMILARITY_THRESHOLD", raising=False)
        monkeypatch.delenv("FACE_IDENTITY_VIDEO_SIMILARITY_THRESHOLD", raising=False)
        monkeypatch.delenv("FACE_IDENTITY_AMBIGUITY_MARGIN", raising=False)
        monkeypatch.delenv("FACE_IDENTITY_MODEL_ID", raising=False)
        monkeypatch.delenv("FACE_IDENTITY_ARCFACE_MODEL_NAME", raising=False)
        monkeypatch.delenv("FACE_IDENTITY_ARCFACE_PROVIDER_ORDER", raising=False)
        monkeypatch.delenv(
            "FACE_IDENTITY_ARCFACE_FALLBACK_BEHAVIOR", raising=False
        )

        settings = Settings.from_env(autoload_dotenv=False)

        assert settings.face_identity_scene_similarity_threshold == 0.68
        assert settings.face_identity_video_similarity_threshold == 0.74
        assert settings.face_identity_ambiguity_margin == 0.03
        assert settings.face_identity_model_id == "edgeface_s_gamma_05"
        assert settings.face_identity_arcface_model_name == "buffalo_l"
        assert settings.face_identity_arcface_provider_order == (
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        )
        assert settings.face_identity_arcface_fallback_behavior == "cpu"

    def test_invalid_face_identity_model_id_fails_fast(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FACE_IDENTITY_PIPELINE", "true")
        monkeypatch.setenv("FACE_IDENTITY_MODEL_ID", "not-a-real-profile")

        with pytest.raises(ValueError, match="Unsupported FACE_IDENTITY_MODEL_ID"):
            Settings.from_env(autoload_dotenv=False)

    def test_invalid_face_identity_model_id_is_ignored_when_pipeline_disabled(
        self, monkeypatch
    ):
        monkeypatch.setenv("ENABLE_FACE_IDENTITY_PIPELINE", "false")
        monkeypatch.setenv("FACE_IDENTITY_MODEL_ID", "not-a-real-profile")

        settings = Settings.from_env(autoload_dotenv=False)

        assert settings.face_identity_model_id == "edgeface_s_gamma_05"

    def test_missing_llm_fields_only_when_pipeline_enabled(self, monkeypatch):
        monkeypatch.setenv("KG_PIPELINE_ENABLED", "true")
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

    def test_missing_face_identity_fields_only_when_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FACE_IDENTITY_PIPELINE", "false")
        settings = Settings.from_env(autoload_dotenv=False)
        assert settings.missing_face_identity_fields() == []

        monkeypatch.setenv("ENABLE_FACE_IDENTITY_PIPELINE", "true")
        settings = Settings.from_env(autoload_dotenv=False)
        object.__setattr__(settings, "face_identity_model_id", "")
        assert settings.missing_face_identity_fields() == ["FACE_IDENTITY_MODEL_ID"]

    def test_autoloads_google_api_key_from_dotenv_local(
        self, monkeypatch, tmp_path: Path
    ):
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
