"""FastAPI application for video analysis API."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
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


def _materialize_signed_result_urls(result_payload: dict[str, Any], media_store: MediaStore) -> dict[str, Any]:
    """Convert stored object keys to signed URLs for API responses."""
    payload: dict[str, Any] = {
        "job_id": result_payload.get("job_id"),
        "frames": [],
    }
    for frame in result_payload.get("frames", []):
        files: dict[str, str] = {}
        raw_files = frame.get("files", {})
        for name, value in raw_files.items():
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                files[name] = value
            elif isinstance(value, str) and value.startswith("jobs/"):
                files[name] = media_store.sign_read_url(value, expires_in=SETTINGS.r2_url_ttl_seconds)
            elif isinstance(value, str):
                # Backward compatibility for older payload formats.
                files[name] = value

        payload["frames"].append(
            {
                "frame_id": frame.get("frame_id"),
                "timestamp": frame.get("timestamp"),
                "files": files,
                "analysis": frame.get("analysis", {}),
            }
        )
    return payload


def process_video(job_id: str, video_path: str, upload_content_type: str | None = None) -> None:
    """Background task: full pipeline from video to results."""
    try:
        media_store = get_media_store()
        media_store.upload_source_video(
            job_id=job_id,
            file_path=video_path,
            content_type=upload_content_type or "video/mp4",
        )

        models = ModelLoader.get()
        scenes = scene.detect_scenes(video_path)
        frames = scene.extract_keyframes(video_path, scenes)
        if not frames:
            jobs.fail_job(job_id, "No scenes or frames extracted")
            return

        scene.save_original_frames(frames, job_id, str(TEMP_MEDIA_DIR), media_store=media_store)

        frame_results = []
        for frame_data in frames:
            result = analysis.analyze_frame(
                frame_data,
                models,
                job_id,
                str(TEMP_MEDIA_DIR),
                media_store=media_store,
            )
            frame_results.append(result)

        payload = {"job_id": job_id, "frames": frame_results}
        jobs.complete_job(job_id, payload)
    except (MediaStoreConfigError, MediaStoreError) as e:
        logger.exception("Media storage failed for job %s", job_id)
        jobs.fail_job(job_id, str(e))
    except Exception as e:
        logger.exception("Video processing failed for job %s", job_id)
        jobs.fail_job(job_id, str(e))
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


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
):
    """Accept video upload, return job_id immediately, process in background."""
    if not file.filename:
        raise HTTPException(422, "No file provided")

    # Stream to temp file with size check
    size = 0
    fd, path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    os.unlink(path)
                    raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise HTTPException(500, str(e))

    job_id = jobs.create_job()
    background_tasks.add_task(process_video, job_id, path, file.content_type)
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
        raise HTTPException(409, "Job is still processing")
    if job["status"] == "failed":
        raise HTTPException(409, f"Job failed: {job.get('error', 'Unknown error')}")
    try:
        payload = _materialize_signed_result_urls(job["result"], get_media_store())
    except (MediaStoreConfigError, MediaStoreError) as exc:
        raise HTTPException(500, str(exc)) from exc
    return JobResult(**payload)
