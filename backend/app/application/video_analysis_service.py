"""Application service boundary for modular video analysis workflow."""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app import cleanup
from app.config import Settings
from app.infrastructure import InMemoryJobStore
from app.models import ModelLoader
from app.pipeline import (
    PipelineContext,
    PipelineExecutionResult,
    PipelineExecutor,
    PipelineStageRegistry,
    build_default_registry,
    build_pipeline_metadata,
    build_result_payload,
)
from app.storage import (
    MediaStore,
    MediaStoreConfigError,
    MediaStoreError,
    build_source_video_key,
)

logger = logging.getLogger(__name__)


class PipelineExecutionError(RuntimeError):
    """Raised when modular pipeline execution fails in one stage."""

    def __init__(
        self,
        message: str,
        *,
        failed_stage: str | None,
        pipeline_metadata: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.failed_stage = failed_stage
        self.pipeline_metadata = pipeline_metadata


def extract_source_extension(filename: str | None) -> str:
    """Extract normalized source extension token without leading dot."""
    if not filename:
        return "mp4"
    extension = Path(filename).suffix.strip().lower().lstrip(".")
    if not extension:
        return "mp4"
    normalized = "".join(ch for ch in extension if ch.isalnum())
    return normalized or "mp4"


def resolve_video_content_type(
    upload_content_type: str | None, source_extension: str
) -> str:
    """Resolve source video content type for object-storage metadata."""
    if upload_content_type and "/" in upload_content_type:
        return upload_content_type
    guessed, _ = mimetypes.guess_type(f"source.{source_extension}")
    if guessed and guessed.startswith("video/"):
        return guessed
    return "video/mp4"


def resolve_cleanup_policy(settings: Settings, request_override: bool | None) -> bool:
    """Resolve effective local-source cleanup policy for one request."""
    if request_override is None:
        return settings.cleanup_local_video_after_upload_default
    return request_override


def build_local_source_path(
    *,
    temp_media_dir: Path,
    job_id: str,
    source_extension: str,
) -> Path:
    """Build deterministic local staging path for uploaded source video."""
    return temp_media_dir / job_id / "input" / f"source.{source_extension}"


def _to_signed_url_if_needed(value: Any, media_store: MediaStore, ttl_seconds: int) -> str:
    if not isinstance(value, str):
        return ""
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("jobs/"):
        return media_store.sign_read_url(value, expires_in=ttl_seconds)
    return value


def _materialize_signed_file_map(
    raw_files: Any,
    media_store: MediaStore,
    ttl_seconds: int,
) -> dict[str, str]:
    files: dict[str, str] = {}
    if not isinstance(raw_files, dict):
        return files
    for name, value in raw_files.items():
        signed_value = _to_signed_url_if_needed(value, media_store, ttl_seconds)
        if signed_value:
            files[name] = signed_value
    return files


def _normalize_object_detection(
    raw_analysis: dict[str, Any],
    frame_id: int,
) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    raw_items = raw_analysis.get("object_detection", [])
    if not isinstance(raw_items, list):
        return normalized_items
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized["track_id"] = (
            normalized.get("track_id") or f"track_{frame_id}_{index + 1}"
        )
        normalized_items.append(normalized)
    return normalized_items


def _normalize_face_recognition(raw_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    raw_items = raw_analysis.get("face_recognition", [])
    if not isinstance(raw_items, list):
        return normalized_items
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized["identity_id"] = normalized.get("identity_id") or f"face_{index + 1}"
        normalized_items.append(normalized)
    return normalized_items


def _default_frame_metadata(
    *,
    job_id: Any,
    frame_id: int,
    timestamp: str,
    raw_frame_index: int | None,
    source_artifact_key: str,
) -> dict[str, Any]:
    resolved_job_id = "" if job_id is None else job_id
    return {
        "provenance": {
            "job_id": str(resolved_job_id),
            "scene_id": None,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "raw_frame_index": raw_frame_index,
            "source_artifact_key": source_artifact_key,
        },
        "model_provenance": [],
        "evidence_anchors": [],
        "extensions": {},
    }


def _materialize_frame_result(
    *,
    frame: dict[str, Any],
    job_id: Any,
    media_store: MediaStore,
    ttl_seconds: int,
) -> dict[str, Any]:
    files = _materialize_signed_file_map(frame.get("files"), media_store, ttl_seconds)
    raw_artifacts = frame.get("analysis_artifacts", {})
    if not isinstance(raw_artifacts, dict):
        raw_artifacts = {}

    artifacts_extensions = raw_artifacts.get("extensions")
    if not isinstance(artifacts_extensions, dict):
        artifacts_extensions = {}
    normalized_artifacts_extensions: dict[str, Any] = {}
    for stage_id, values in artifacts_extensions.items():
        if isinstance(values, dict):
            normalized_artifacts_extensions[str(stage_id)] = {
                key: _to_signed_url_if_needed(value, media_store, ttl_seconds)
                for key, value in values.items()
            }

    analysis_artifacts = {
        "json": _to_signed_url_if_needed(raw_artifacts.get("json"), media_store, ttl_seconds),
        "extensions": normalized_artifacts_extensions,
    }

    frame_id = int(frame.get("frame_id", 0))
    timestamp = str(frame.get("timestamp", ""))
    raw_frame_index_value = frame.get("raw_frame_index")
    raw_frame_index = (
        raw_frame_index_value
        if isinstance(raw_frame_index_value, int)
        and not isinstance(raw_frame_index_value, bool)
        else None
    )
    raw_analysis = frame.get("analysis", {})
    if not isinstance(raw_analysis, dict):
        raw_analysis = {}

    analysis_extensions = raw_analysis.get("extensions")
    if not isinstance(analysis_extensions, dict):
        analysis_extensions = {}
    normalized_analysis_extensions = {
        str(stage_id): dict(values)
        for stage_id, values in analysis_extensions.items()
        if isinstance(values, dict)
    }

    metadata = frame.get("metadata")
    if not isinstance(metadata, dict):
        metadata = _default_frame_metadata(
            job_id=job_id,
            frame_id=frame_id,
            timestamp=timestamp,
            raw_frame_index=raw_frame_index,
            source_artifact_key=files.get("original", ""),
        )
    provenance = metadata.get("provenance")
    if isinstance(provenance, dict):
        provenance.setdefault("raw_frame_index", raw_frame_index)
    metadata.setdefault("extensions", {})

    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "raw_frame_index": raw_frame_index,
        "files": files,
        "analysis": {
            "semantic_segmentation": raw_analysis.get("semantic_segmentation", []),
            "object_detection": _normalize_object_detection(raw_analysis, frame_id),
            "face_recognition": _normalize_face_recognition(raw_analysis),
            "enrichment": raw_analysis.get("enrichment", {}),
            "extensions": normalized_analysis_extensions,
        },
        "analysis_artifacts": analysis_artifacts,
        "metadata": metadata,
    }


def materialize_signed_result_urls(
    *,
    result_payload: dict[str, Any],
    media_store: MediaStore,
    ttl_seconds: int,
    fallback_pipeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert stored artifact keys to signed URLs for API responses."""
    payload: dict[str, Any] = {
        "job_id": result_payload.get("job_id"),
        "pipeline": result_payload.get("pipeline")
        if isinstance(result_payload.get("pipeline"), dict)
        else (fallback_pipeline or {"stages": [], "status": [], "failed_stage": None}),
        "branch_metadata": result_payload.get("branch_metadata")
        if isinstance(result_payload.get("branch_metadata"), dict)
        else None,
        "frames": [],
    }
    raw_chunked_tracks = result_payload.get("video_chunked_tracks")
    if isinstance(raw_chunked_tracks, dict):
        payload["video_chunked_tracks"] = dict(raw_chunked_tracks)
    raw_face_identities = result_payload.get("video_face_identities")
    if isinstance(raw_face_identities, dict):
        payload["video_face_identities"] = dict(raw_face_identities)
    raw_object_tracks = result_payload.get("video_object_tracks")
    if isinstance(raw_object_tracks, dict):
        payload["video_object_tracks"] = dict(raw_object_tracks)
    raw_person_tracks = result_payload.get("video_person_tracks")
    if isinstance(raw_person_tracks, dict):
        payload["video_person_tracks"] = dict(raw_person_tracks)

    raw_frames = result_payload.get("frames", [])
    if not isinstance(raw_frames, list):
        raw_frames = []

    materialized_frames = []
    for frame in raw_frames:
        if not isinstance(frame, dict):
            continue
        materialized_frames.append(
            _materialize_frame_result(
                frame=frame,
                job_id=result_payload.get("job_id"),
                media_store=media_store,
                ttl_seconds=ttl_seconds,
            )
        )

    payload["frames"] = sorted(
        materialized_frames, key=lambda item: int(item.get("frame_id", 0))
    )
    return payload


def finalize_local_source_video(
    *,
    temp_media_dir: Path,
    job_id: str,
    video_path: str,
    cleanup_after_upload: bool,
    source_upload_verified: bool,
) -> None:
    """Finalize local source file handling based on verification and policy."""
    source_path = Path(video_path)
    if cleanup_after_upload and source_upload_verified:
        try:
            source_path.unlink()
            cleanup.clear_job_source_retention_marker(str(temp_media_dir), job_id)
            logger.info(
                "cleanup.local_source_deleted job_id=%s path=%s", job_id, source_path
            )
        except OSError:
            logger.warning(
                "cleanup.local_source_delete_failed job_id=%s path=%s",
                job_id,
                source_path,
            )
        return

    if not cleanup_after_upload and source_upload_verified:
        cleanup.mark_job_for_source_retention(str(temp_media_dir), job_id)
        logger.info(
            "cleanup.local_source_retained job_id=%s path=%s", job_id, source_path
        )
        return

    logger.info(
        "cleanup.local_source_preserved_unverified job_id=%s path=%s",
        job_id,
        source_path,
    )


@dataclass(slots=True)
class VideoAnalysisService:
    """Application service for submit/status/results and pipeline execution."""

    settings: Settings
    temp_media_dir: Path
    media_store_provider: Callable[[], MediaStore]
    job_store: InMemoryJobStore
    model_loader_cls: type[ModelLoader] = ModelLoader
    registry_factory: Callable[[], PipelineStageRegistry] = build_default_registry

    def submit_job(
        self,
        *,
        background_tasks: Any,
        job_id: str,
        local_source_path: str,
        source_extension: str,
        upload_content_type: str | None,
        cleanup_local_video_after_upload: bool | None,
    ) -> str:
        """Create queued job state and schedule background processing."""
        effective_cleanup_policy = resolve_cleanup_policy(
            self.settings, cleanup_local_video_after_upload
        )
        self.job_store.create(
            job_id=job_id,
            status="queued",
            metadata={
                "cleanup_local_video_after_upload": effective_cleanup_policy,
                "local_source_path": str(local_source_path),
                "source_extension": source_extension,
            },
        )
        background_tasks.add_task(
            self.run_pipeline_job,
            job_id,
            str(local_source_path),
            source_extension,
            upload_content_type,
            cleanup_local_video_after_upload,
        )
        return job_id

    def _run_modular_pipeline(
        self,
        *,
        job_id: str,
        video_path: str,
        source_extension: str,
        upload_content_type: str | None,
        media_store: MediaStore,
    ) -> tuple[PipelineExecutionResult, dict[str, Any]]:
        context = PipelineContext(
            job_id=job_id,
            video_path=video_path,
            source_extension=source_extension,
            upload_content_type=upload_content_type,
            local_dir=str(self.temp_media_dir),
            settings=self.settings,
            models=self.model_loader_cls.get(),
            media_store=media_store,
        )
        executor = PipelineExecutor()
        execution = executor.execute(
            context=context,
            registry=self.registry_factory(),
            on_stage_started=lambda stage_id: self.job_store.set_stage(job_id, stage_id),
        )
        pipeline_metadata = build_pipeline_metadata(execution)
        if not execution.is_success:
            failed_stage = execution.failed_stage_id
            message = execution.failure.error if execution.failure else "Pipeline failed"
            raise PipelineExecutionError(
                message,
                failed_stage=failed_stage,
                pipeline_metadata=pipeline_metadata,
            )
        return execution, build_result_payload(job_id=job_id, execution=execution)

    def run_pipeline_job(
        self,
        job_id: str,
        video_path: str,
        source_extension: str = "mp4",
        upload_content_type: str | None = None,
        cleanup_local_video_after_upload: bool | None = None,
    ) -> None:
        """Execute pipeline job lifecycle in background worker context."""
        cleanup_after_upload = resolve_cleanup_policy(
            self.settings, cleanup_local_video_after_upload
        )
        source_upload_verified = False
        source_key: str | None = None
        pipeline_metadata: dict[str, Any] | None = None
        media_store: MediaStore | None = None

        self.job_store.set_processing(job_id, current_stage="pipeline_start")
        try:
            media_store = self.media_store_provider()
            execution, payload = self._run_modular_pipeline(
                job_id=job_id,
                video_path=video_path,
                source_extension=source_extension,
                upload_content_type=upload_content_type,
                media_store=media_store,
            )
            pipeline_metadata = build_pipeline_metadata(execution)
            source_key = execution.stage_outputs.get("source_upload", {}).get(
                "source_key"
            )

            if source_key:
                source_upload_verified = media_store.verify_object(source_key)
            self.job_store.complete(job_id, payload, pipeline=pipeline_metadata)
        except PipelineExecutionError as exc:
            logger.exception("Pipeline execution failed for job %s", job_id)
            self.job_store.fail(
                job_id,
                str(exc),
                failed_stage=exc.failed_stage,
                pipeline=exc.pipeline_metadata,
            )
        except (MediaStoreConfigError, MediaStoreError) as exc:
            logger.exception("Media storage failed for job %s", job_id)
            self.job_store.fail(
                job_id,
                str(exc),
                failed_stage=None,
                pipeline=pipeline_metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Video processing failed for job %s", job_id)
            self.job_store.fail(
                job_id,
                str(exc),
                failed_stage=None,
                pipeline=pipeline_metadata,
            )
        finally:
            if (
                not source_upload_verified
                and media_store is not None
                and source_key is None
            ):
                fallback_source_key = build_source_video_key(
                    job_id,
                    source_extension=source_extension,
                )
                source_upload_verified = media_store.verify_object(fallback_source_key)

            finalize_local_source_video(
                temp_media_dir=self.temp_media_dir,
                job_id=job_id,
                video_path=video_path,
                cleanup_after_upload=cleanup_after_upload,
                source_upload_verified=source_upload_verified,
            )

    def get_status_payload(self, job_id: str) -> dict[str, Any] | None:
        """Build public status payload for a job."""
        job = self.job_store.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job_id,
            "status": job.get("status", "processing"),
            "error": job.get("error"),
            "current_stage": job.get("current_stage") or job.get("stage"),
            "failed_stage": job.get("failed_stage"),
        }

    def get_result_payload(self, job_id: str) -> dict[str, Any] | None:
        """Build signed result payload for completed jobs."""
        job = self.job_store.get(job_id)
        if job is None:
            return None
        if job.get("status") == "processing" or job.get("status") == "queued":
            return {"detail": "Job is still processing", "status_code": 409}
        if job.get("status") == "failed":
            return {
                "detail": "Job failed",
                "error": job.get("error", "Unknown error"),
                "failed_stage": job.get("failed_stage"),
                "status_code": 409,
            }
        result_payload = job.get("result", {})
        return materialize_signed_result_urls(
            result_payload=result_payload,
            media_store=self.media_store_provider(),
            ttl_seconds=self.settings.r2_url_ttl_seconds,
            fallback_pipeline=job.get("pipeline"),
        )
