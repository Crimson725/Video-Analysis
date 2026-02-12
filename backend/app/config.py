"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _read_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _load_dotenv_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a dotenv file without overriding existing env."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _load_dotenv_files(paths: Iterable[Path]) -> None:
    """Load multiple dotenv files in order."""
    for path in paths:
        _load_dotenv_file(path)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for API and media storage."""

    r2_account_id: str
    r2_bucket: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_url_ttl_seconds: int
    r2_retention_days: int
    r2_abort_multipart_days: int
    temp_media_dir: str
    cleanup_local_video_after_upload_default: bool
    enable_scene_understanding_pipeline: bool
    google_api_key: str
    scene_model_id: str
    synopsis_model_id: str
    scene_llm_retry_count: int
    scene_packet_max_entities: int
    scene_packet_max_events: int
    scene_packet_max_keyframes: int
    scene_packet_disambiguation_label_threshold: int
    langsmith_tracing_enabled: bool
    langsmith_project: str
    enable_corpus_pipeline: bool
    enable_corpus_ingest: bool
    graph_backend: str
    vector_backend: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    pgvector_dsn: str
    embedding_model_id: str
    embedding_model_version: str
    embedding_dimension: int

    @classmethod
    def from_env(
        cls,
        *,
        autoload_dotenv: bool = True,
        dotenv_files: tuple[Path, ...] | None = None,
    ) -> "Settings":
        default_temp = Path(__file__).resolve().parent.parent / "tmp_media"
        if autoload_dotenv:
            backend_root = Path(__file__).resolve().parent.parent
            default_dotenvs = (backend_root / ".env", backend_root / ".env.local")
            _load_dotenv_files(dotenv_files or default_dotenvs)
        return cls(
            r2_account_id=os.getenv("R2_ACCOUNT_ID", "").strip(),
            r2_bucket=os.getenv("R2_BUCKET", "").strip(),
            r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            r2_url_ttl_seconds=_read_int("R2_URL_TTL_SECONDS", default=3600, minimum=1),
            r2_retention_days=_read_int("R2_RETENTION_DAYS", default=7, minimum=1),
            r2_abort_multipart_days=_read_int("R2_ABORT_MULTIPART_DAYS", default=1, minimum=1),
            temp_media_dir=os.getenv("TEMP_MEDIA_DIR", str(default_temp)).strip(),
            cleanup_local_video_after_upload_default=_read_bool(
                "CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT",
                default=True,
            ),
            enable_scene_understanding_pipeline=_read_bool(
                "ENABLE_SCENE_UNDERSTANDING_PIPELINE",
                default=True,
            ),
            google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
            scene_model_id=os.getenv("SCENE_MODEL_ID", "gemini-2.5-flash-lite").strip(),
            synopsis_model_id=os.getenv("SYNOPSIS_MODEL_ID", "gemini-2.5-flash-lite").strip(),
            scene_llm_retry_count=_read_int("SCENE_LLM_RETRY_COUNT", default=2, minimum=0),
            scene_packet_max_entities=_read_int("SCENE_PACKET_MAX_ENTITIES", default=8, minimum=1),
            scene_packet_max_events=_read_int("SCENE_PACKET_MAX_EVENTS", default=8, minimum=1),
            scene_packet_max_keyframes=_read_int("SCENE_PACKET_MAX_KEYFRAMES", default=3, minimum=1),
            scene_packet_disambiguation_label_threshold=_read_int(
                "SCENE_PACKET_DISAMBIGUATION_LABEL_THRESHOLD",
                default=4,
                minimum=1,
            ),
            langsmith_tracing_enabled=_read_bool("LANGSMITH_TRACING", default=False),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "").strip(),
            enable_corpus_pipeline=_read_bool("ENABLE_CORPUS_PIPELINE", default=True),
            enable_corpus_ingest=_read_bool("ENABLE_CORPUS_INGEST", default=True),
            graph_backend=os.getenv("GRAPH_BACKEND", "neo4j").strip() or "neo4j",
            vector_backend=os.getenv("VECTOR_BACKEND", "pgvector").strip() or "pgvector",
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687").strip(),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j").strip(),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "local-dev-password").strip(),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j").strip(),
            pgvector_dsn=os.getenv(
                "PGVECTOR_DSN",
                "postgresql://video_analysis:video_analysis@127.0.0.1:5433/video_analysis",
            ).strip(),
            embedding_model_id=os.getenv("EMBEDDING_MODEL_ID", "local-hash-embedding").strip(),
            embedding_model_version=os.getenv("EMBEDDING_MODEL_VERSION", "v1").strip(),
            embedding_dimension=_read_int("EMBEDDING_DIMENSION", default=16, minimum=1),
        )

    def missing_r2_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.r2_account_id:
            missing.append("R2_ACCOUNT_ID")
        if not self.r2_bucket:
            missing.append("R2_BUCKET")
        if not self.r2_access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.r2_secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        return missing

    def missing_llm_fields(self) -> list[str]:
        """Return missing required LLM settings for enabled scene understanding."""
        missing: list[str] = []
        if self.enable_scene_understanding_pipeline and not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        return missing
