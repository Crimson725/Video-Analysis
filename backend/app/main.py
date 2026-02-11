"""FastAPI application for video analysis API."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import analysis, cleanup, jobs, scene
from app.models import ModelLoader
from app.schemas import JobResult, JobStatus

logger = logging.getLogger(__name__)

# Config
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def process_video(job_id: str, video_path: str) -> None:
    """Background task: full pipeline from video to results."""
    try:
        models = ModelLoader.get()
        scenes = scene.detect_scenes(video_path)
        frames = scene.extract_keyframes(video_path, scenes)
        if not frames:
            jobs.fail_job(job_id, "No scenes or frames extracted")
            return

        scene.save_original_frames(frames, job_id, str(STATIC_DIR))

        frame_results = []
        for frame_data in frames:
            result = analysis.analyze_frame(frame_data, models, job_id, str(STATIC_DIR))
            frame_results.append(result)

        payload = {"job_id": job_id, "frames": frame_results}
        jobs.complete_job(job_id, payload)
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
    ModelLoader.get()
    cleanup.setup_scheduler(str(STATIC_DIR))
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

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
    background_tasks.add_task(process_video, job_id, path)
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
    return JobResult(**job["result"])
