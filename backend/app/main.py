"""FastAPI application for modular video analysis API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import analysis, cleanup, jobs, scene
from app.api import stream_upload_to_local_file
from app.application import (
    VideoAnalysisService,
    build_local_source_path,
    extract_source_extension,
    materialize_signed_result_urls,
    resolve_video_content_type,
)
from app.config import Settings
from app.infrastructure import InMemoryJobStore
from app.models import ModelLoader
from app.pipeline.artifacts import (
    collect_required_artifact_keys,
    verify_required_artifacts,
)
from app.schemas import JobResult, JobStatus
from app.storage import MediaStore, MediaStoreConfigError, MediaStoreError, R2MediaStore

logger = logging.getLogger(__name__)

SETTINGS = Settings.from_env()
TEMP_MEDIA_DIR = Path(SETTINGS.temp_media_dir)
TEMP_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
_media_store: MediaStore | None = None


def get_media_store() -> MediaStore:
    """Build and cache the R2 media store instance."""
    global _media_store
    if _media_store is None:
        _media_store = R2MediaStore(
            account_id=SETTINGS.r2_account_id,
            bucket=SETTINGS.r2_bucket,
            access_key_id=SETTINGS.r2_access_key_id,
            secret_access_key=SETTINGS.r2_secret_access_key,
            default_url_ttl_seconds=SETTINGS.r2_url_ttl_seconds,
        )
    return _media_store


def _startup_validate_settings() -> None:
    missing = SETTINGS.missing_r2_fields()
    if missing:
        logger.warning(
            "Missing R2 configuration at startup: %s. "
            "Video processing and signed result URLs will fail until configured.",
            ", ".join(missing),
        )


def _build_service() -> VideoAnalysisService:
    return VideoAnalysisService(
        settings=SETTINGS,
        temp_media_dir=TEMP_MEDIA_DIR,
        media_store_provider=get_media_store,
        job_store=InMemoryJobStore(),
        model_loader_cls=ModelLoader,
    )


def _collect_required_artifact_keys(
    job_id: str,
    result_payload: dict[str, Any],
    source_key: str,
) -> set[str]:
    """Compatibility wrapper for tests using legacy helper import."""
    return collect_required_artifact_keys(job_id, result_payload, source_key)


def _verify_required_artifacts(
    media_store: MediaStore,
    job_id: str,
    required_keys: set[str],
) -> None:
    """Compatibility wrapper for tests using legacy helper import."""
    verify_required_artifacts(media_store, job_id, required_keys)


def _materialize_signed_result_urls(
    result_payload: dict[str, Any],
    media_store: MediaStore,
) -> dict[str, Any]:
    """Compatibility wrapper for tests using legacy helper import."""
    return materialize_signed_result_urls(
        result_payload=result_payload,
        media_store=media_store,
        ttl_seconds=SETTINGS.r2_url_ttl_seconds,
        fallback_pipeline=result_payload.get("pipeline")
        if isinstance(result_payload.get("pipeline"), dict)
        else None,
    )


def process_video_legacy(
    *,
    job_id: str,
    video_path: str,
    source_extension: str = "mp4",
    upload_content_type: str | None = None,
    media_store: MediaStore,
    model_loader_cls: type[ModelLoader],
    temp_media_dir: Path,
    on_stage_started: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Legacy monolithic orchestration kept as temporary rollback path."""
    if on_stage_started is not None:
        on_stage_started("source_upload")
    source_key = media_store.upload_source_video(
        job_id=job_id,
        file_path=video_path,
        content_type=resolve_video_content_type(upload_content_type, source_extension),
        source_extension=source_extension,
    )
    source_upload_verified = media_store.verify_object(source_key)
    if not source_upload_verified:
        logger.error("upload.verify.source_failed job_id=%s key=%s", job_id, source_key)
        raise MediaStoreError(f"Source upload verification failed for key '{source_key}'")

    if on_stage_started is not None:
        on_stage_started("scene_detection")
    models = model_loader_cls.get()
    scenes = scene.detect_scenes(video_path)

    if on_stage_started is not None:
        on_stage_started("keyframe_extraction")
    frames = scene.extract_keyframes(video_path, scenes)
    if not frames:
        raise RuntimeError("No scenes or frames extracted")

    if on_stage_started is not None:
        on_stage_started("save_original_frames")
    scene.save_original_frames(
        frames, job_id, str(temp_media_dir), media_store=media_store
    )

    if on_stage_started is not None:
        on_stage_started("frame_analysis")
    frame_results: list[dict[str, Any]] = []
    face_tracker = analysis.FaceIdentityTracker()
    object_tracker = analysis.ObjectTrackTracker()
    for frame_data in frames:
        result = analysis.analyze_frame(
            frame_data,
            models,
            job_id,
            str(temp_media_dir),
            media_store=media_store,
            face_tracker=face_tracker,
            object_tracker=object_tracker,
        )
        frame_results.append(result)

    payload = {
        "job_id": job_id,
        "frames": frame_results,
    }

    if on_stage_started is not None:
        on_stage_started("artifact_verification")
    required_keys = _collect_required_artifact_keys(job_id, payload, source_key)
    _verify_required_artifacts(media_store, job_id, required_keys)
    payload["_source_key"] = source_key
    return payload


def process_video(
    job_id: str,
    video_path: str,
    source_extension: str = "mp4",
    upload_content_type: str | None = None,
    cleanup_local_video_after_upload: bool | None = None,
) -> None:
    """Background task wrapper maintained for compatibility with tests/callers."""
    _build_service().run_pipeline_job(
        job_id=job_id,
        video_path=video_path,
        source_extension=source_extension,
        upload_content_type=upload_content_type,
        cleanup_local_video_after_upload=cleanup_local_video_after_upload,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, start scheduler. Shutdown: stop scheduler."""
    _startup_validate_settings()
    ModelLoader.get()
    cleanup.setup_scheduler(str(TEMP_MEDIA_DIR))
    yield
    cleanup.shutdown_scheduler()


app = FastAPI(title="Video Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    cleanup_local_video_after_upload: bool | None = Form(default=None),
):
    """Accept upload, persist queued job, and schedule background processing."""
    if not file.filename:
        raise HTTPException(422, "No file provided")

    job_id = str(uuid4())
    source_extension = extract_source_extension(file.filename)
    local_source_path = build_local_source_path(
        temp_media_dir=TEMP_MEDIA_DIR,
        job_id=job_id,
        source_extension=source_extension,
    )
    await stream_upload_to_local_file(
        file=file,
        destination=local_source_path,
        max_upload_bytes=MAX_UPLOAD_BYTES,
        cleanup_root=TEMP_MEDIA_DIR,
        job_id=job_id,
    )

    service = _build_service()
    service.submit_job(
        background_tasks=background_tasks,
        job_id=job_id,
        local_source_path=str(local_source_path),
        source_extension=source_extension,
        upload_content_type=file.content_type,
        cleanup_local_video_after_upload=cleanup_local_video_after_upload,
    )
    logger.info(
        "upload.accepted job_id=%s local_source_path=%s cleanup_after_upload=%s",
        job_id,
        local_source_path,
        jobs.get_job(job_id).get("cleanup_local_video_after_upload"),  # type: ignore[union-attr]
    )
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Return job lifecycle status with stage progress metadata."""
    payload = _build_service().get_status_payload(job_id)
    if payload is None:
        raise HTTPException(404, "Job not found")
    return JobStatus(**payload)


@app.get("/results/{job_id}", response_model=JobResult)
async def get_results(job_id: str):
    """Return signed frame results with top-level pipeline metadata."""
    try:
        payload = _build_service().get_result_payload(job_id)
    except (MediaStoreConfigError, MediaStoreError) as exc:
        raise HTTPException(500, str(exc)) from exc

    if payload is None:
        raise HTTPException(404, "Job not found")
    status_code = payload.pop("status_code", None)
    if status_code:
        return JSONResponse(payload, status_code=status_code)
    return JobResult(**payload)
