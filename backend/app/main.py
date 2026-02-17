"""FastAPI application for video analysis API."""

import logging
import mimetypes
import shutil
from collections.abc import Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import analysis, cleanup, jobs, scene
from app.config import Settings
from app.models import ModelLoader
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
    """Log missing R2 settings during startup validation."""
    missing = SETTINGS.missing_r2_fields()
    if missing:
        logger.warning(
            "Missing R2 configuration at startup: %s. "
            "Video processing and signed result URLs will fail until configured.",
            ", ".join(missing),
        )


def _extract_source_extension(filename: str | None) -> str:
    """Extract normalized source extension token without leading dot."""
    if not filename:
        return "mp4"
    extension = Path(filename).suffix.strip().lower().lstrip(".")
    if not extension:
        return "mp4"
    normalized = "".join(ch for ch in extension if ch.isalnum())
    return normalized or "mp4"


def _resolve_video_content_type(
    upload_content_type: str | None, source_extension: str
) -> str:
    """Resolve source video content type for R2 metadata."""
    if upload_content_type and "/" in upload_content_type:
        return upload_content_type
    guessed, _ = mimetypes.guess_type(f"source.{source_extension}")
    if guessed and guessed.startswith("video/"):
        return guessed
    return "video/mp4"


def _resolve_cleanup_policy(request_override: bool | None) -> bool:
    """Resolve effective local-source cleanup policy for a request."""
    if request_override is None:
        return SETTINGS.cleanup_local_video_after_upload_default
    return request_override


def _build_local_source_path(job_id: str, source_extension: str) -> Path:
    """Build deterministic local staging path for uploaded source video."""
    return TEMP_MEDIA_DIR / job_id / "input" / f"source.{source_extension}"


def _to_signed_url_if_needed(value: Any, media_store: MediaStore) -> str:
    """Convert object key to signed URL while keeping existing URLs unchanged."""
    if not isinstance(value, str):
        return ""
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("jobs/"):
        return media_store.sign_read_url(value, expires_in=SETTINGS.r2_url_ttl_seconds)
    return value


def _materialize_signed_file_map(
    raw_files: Any, media_store: MediaStore
) -> dict[str, str]:
    """Sign frame file references while dropping invalid/empty values."""
    files: dict[str, str] = {}
    for name, value in raw_files.items():
        signed_value = _to_signed_url_if_needed(value, media_store)
        if signed_value:
            files[name] = signed_value
    return files


def _normalize_object_detection(
    raw_analysis: dict[str, Any], frame_id: int
) -> list[dict[str, Any]]:
    """Normalize object detections and fill missing track identifiers."""
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_analysis.get("object_detection", [])):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized["track_id"] = (
            normalized.get("track_id") or f"track_{frame_id}_{index + 1}"
        )
        normalized_items.append(normalized)
    return normalized_items


def _normalize_face_recognition(raw_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize face detections and fill missing identity identifiers."""
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_analysis.get("face_recognition", [])):
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
    source_artifact_key: str,
) -> dict[str, Any]:
    """Build fallback frame metadata when upstream metadata is missing."""
    resolved_job_id = "" if job_id is None else job_id
    return {
        "provenance": {
            "job_id": str(resolved_job_id),
            "scene_id": None,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "source_artifact_key": source_artifact_key,
        },
        "model_provenance": [],
        "evidence_anchors": [],
    }


def _materialize_frame_result(
    *,
    frame: dict[str, Any],
    job_id: Any,
    media_store: MediaStore,
) -> dict[str, Any]:
    """Materialize one frame payload with signed URLs and normalized analysis fields."""
    files = _materialize_signed_file_map(frame.get("files"), media_store)
    raw_artifacts = frame.get("analysis_artifacts", {})
    analysis_artifacts = {
        "json": _to_signed_url_if_needed(raw_artifacts.get("json"), media_store),
    }
    frame_id = int(frame.get("frame_id", 0))
    timestamp = str(frame.get("timestamp", ""))
    raw_analysis = frame.get("analysis", {})
    object_detection = _normalize_object_detection(raw_analysis, frame_id)
    face_recognition = _normalize_face_recognition(raw_analysis)

    metadata = frame.get("metadata")
    if not isinstance(metadata, dict):
        metadata = _default_frame_metadata(
            job_id=job_id,
            frame_id=frame_id,
            timestamp=timestamp,
            source_artifact_key=files.get("original", ""),
        )

    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "files": files,
        "analysis": {
            "semantic_segmentation": raw_analysis.get("semantic_segmentation", []),
            "object_detection": object_detection,
            "face_recognition": face_recognition,
            "enrichment": raw_analysis.get("enrichment", {}),
        },
        "analysis_artifacts": analysis_artifacts,
        "metadata": metadata,
    }


def _materialize_signed_result_urls(
    result_payload: dict[str, Any], media_store: MediaStore
) -> dict[str, Any]:
    """Convert stored CV artifact keys to signed URLs for API responses."""
    payload: dict[str, Any] = {
        "job_id": result_payload.get("job_id"),
        "frames": [],
    }
    raw_frames = result_payload.get("frames", [])
    if not isinstance(raw_frames, list):
        raw_frames = []

    for frame in raw_frames:
        if not isinstance(frame, dict):
            continue
        payload["frames"].append(
            _materialize_frame_result(
                frame=frame,
                job_id=result_payload.get("job_id"),
                media_store=media_store,
            )
        )
    return payload


def _is_jobs_object_key(object_key: Any) -> bool:
    """Check whether a value is a persisted artifact object key."""
    return isinstance(object_key, str) and object_key.startswith("jobs/")


def _add_required_key_or_log(
    *,
    required: set[str],
    object_key: Any,
    warning_template: str,
    warning_context: tuple[Any, ...],
) -> None:
    """Add valid object keys to required set, otherwise emit existing warning logs."""
    if _is_jobs_object_key(object_key):
        required.add(object_key)
        return
    logger.warning(warning_template, *warning_context, object_key)


def _collect_required_keys_from_values(
    *,
    required: set[str],
    values: Iterable[Any],
    warning_template: str,
    warning_context: tuple[Any, ...],
) -> None:
    """Collect required keys from an iterable while preserving invalid-key warning semantics."""
    for object_key in values:
        _add_required_key_or_log(
            required=required,
            object_key=object_key,
            warning_template=warning_template,
            warning_context=warning_context,
        )


def _collect_required_artifact_keys(
    job_id: str,
    result_payload: dict[str, Any],
    source_key: str,
) -> set[str]:
    """Collect required R2 object keys that must verify before job completion."""
    required: set[str] = {source_key}
    for frame in result_payload.get("frames", []):
        frame_id = frame.get("frame_id")
        _collect_required_keys_from_values(
            required=required,
            values=frame.get("files", {}).values(),
            warning_template="upload.verify.invalid_frame_file_key job_id=%s frame_id=%s value=%s",
            warning_context=(job_id, frame_id),
        )
        _collect_required_keys_from_values(
            required=required,
            values=frame.get("analysis_artifacts", {}).values(),
            warning_template="upload.verify.invalid_analysis_key job_id=%s frame_id=%s value=%s",
            warning_context=(job_id, frame_id),
        )

    return required


def _verify_required_artifacts(
    media_store: MediaStore, job_id: str, required_keys: set[str]
) -> None:
    """Verify required objects exist in R2 before marking a job complete."""
    missing = sorted(key for key in required_keys if not media_store.verify_object(key))
    if missing:
        preview = ", ".join(missing[:5])
        logger.error(
            "upload.verify.failed job_id=%s missing_count=%s sample=%s",
            job_id,
            len(missing),
            preview,
        )
        raise MediaStoreError(
            f"Upload verification failed for {len(missing)} artifact(s); sample: {preview}"
        )
    logger.info(
        "upload.verify.success job_id=%s artifact_count=%s", job_id, len(required_keys)
    )


def _finalize_local_source_video(
    *,
    job_id: str,
    video_path: str,
    cleanup_after_upload: bool,
    source_upload_verified: bool,
) -> None:
    """Finalize local source file handling based on upload verification and policy."""
    source_path = Path(video_path)
    if cleanup_after_upload and source_upload_verified:
        try:
            source_path.unlink()
            cleanup.clear_job_source_retention_marker(str(TEMP_MEDIA_DIR), job_id)
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
        cleanup.mark_job_for_source_retention(str(TEMP_MEDIA_DIR), job_id)
        logger.info(
            "cleanup.local_source_retained job_id=%s path=%s", job_id, source_path
        )
        return

    logger.info(
        "cleanup.local_source_preserved_unverified job_id=%s path=%s",
        job_id,
        source_path,
    )


def process_video(
    job_id: str,
    video_path: str,
    source_extension: str = "mp4",
    upload_content_type: str | None = None,
    cleanup_local_video_after_upload: bool | None = None,
) -> None:
    """Background task: full pipeline from video to results."""
    cleanup_after_upload = _resolve_cleanup_policy(cleanup_local_video_after_upload)
    source_upload_verified = False
    try:
        media_store = get_media_store()
        source_key = media_store.upload_source_video(
            job_id=job_id,
            file_path=video_path,
            content_type=_resolve_video_content_type(
                upload_content_type, source_extension
            ),
            source_extension=source_extension,
        )
        source_upload_verified = media_store.verify_object(source_key)
        if not source_upload_verified:
            logger.error(
                "upload.verify.source_failed job_id=%s key=%s", job_id, source_key
            )
            raise MediaStoreError(
                f"Source upload verification failed for key '{source_key}'"
            )

        jobs.set_job_stage(job_id, "cv_processing")
        models = ModelLoader.get()
        scenes = scene.detect_scenes(video_path)
        frames = scene.extract_keyframes(video_path, scenes)
        if not frames:
            jobs.fail_job(job_id, "No scenes or frames extracted")
            return

        scene.save_original_frames(
            frames, job_id, str(TEMP_MEDIA_DIR), media_store=media_store
        )

        frame_results = []
        face_tracker = analysis.FaceIdentityTracker()
        object_tracker = analysis.ObjectTrackTracker()
        for frame_data in frames:
            result = analysis.analyze_frame(
                frame_data,
                models,
                job_id,
                str(TEMP_MEDIA_DIR),
                media_store=media_store,
                face_tracker=face_tracker,
                object_tracker=object_tracker,
            )
            frame_results.append(result)

        payload = {
            "job_id": job_id,
            "frames": frame_results,
        }
        required_keys = _collect_required_artifact_keys(job_id, payload, source_key)
        _verify_required_artifacts(media_store, job_id, required_keys)

        jobs.complete_job(job_id, payload)
    except (MediaStoreConfigError, MediaStoreError) as e:
        logger.exception("Media storage failed for job %s", job_id)
        jobs.fail_job(job_id, str(e))
    except Exception as e:
        logger.exception("Video processing failed for job %s", job_id)
        jobs.fail_job(job_id, str(e))
    finally:
        _finalize_local_source_video(
            job_id=job_id,
            video_path=video_path,
            cleanup_after_upload=cleanup_after_upload,
            source_upload_verified=source_upload_verified,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, start scheduler. Shutdown: stop scheduler."""
    _startup_validate_settings()
    ModelLoader.get()
    cleanup.setup_scheduler(str(TEMP_MEDIA_DIR))
    yield
    cleanup.shutdown_scheduler()


app = FastAPI(
    title="Video Analysis API",
    lifespan=lifespan,
)

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
    """Accept video upload, return job_id immediately, process in background."""
    if not file.filename:
        raise HTTPException(422, "No file provided")

    job_id = str(uuid4())
    source_extension = _extract_source_extension(file.filename)
    local_source_path = _build_local_source_path(job_id, source_extension)
    local_source_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream to deterministic local staged file with size check.
    size = 0
    try:
        with local_source_path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    local_source_path.unlink(missing_ok=True)
                    shutil.rmtree(TEMP_MEDIA_DIR / job_id, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        local_source_path.unlink(missing_ok=True)
        shutil.rmtree(TEMP_MEDIA_DIR / job_id, ignore_errors=True)
        raise HTTPException(500, str(e))

    effective_cleanup_policy = _resolve_cleanup_policy(cleanup_local_video_after_upload)
    jobs.create_job(
        job_id=job_id,
        metadata={
            "cleanup_local_video_after_upload": effective_cleanup_policy,
            "local_source_path": str(local_source_path),
            "source_extension": source_extension,
        },
    )
    logger.info(
        "upload.accepted job_id=%s local_source_path=%s cleanup_after_upload=%s",
        job_id,
        local_source_path,
        effective_cleanup_policy,
    )
    background_tasks.add_task(
        process_video,
        job_id,
        str(local_source_path),
        source_extension,
        file.content_type,
        cleanup_local_video_after_upload,
    )
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Return job status (processing, completed, failed)."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
    )


@app.get("/results/{job_id}", response_model=JobResult)
async def get_results(job_id: str):
    """Return full analysis JSON when completed."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] == "processing":
        return JSONResponse({"detail": "Job is still processing"}, status_code=409)
    if job["status"] == "failed":
        return JSONResponse(
            {"detail": "Job failed", "error": job.get("error", "Unknown error")},
            status_code=409,
        )
    try:
        payload = _materialize_signed_result_urls(job["result"], get_media_store())
    except (MediaStoreConfigError, MediaStoreError) as exc:
        raise HTTPException(500, str(exc)) from exc
    return JobResult(**payload)
