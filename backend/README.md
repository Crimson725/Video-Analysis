# Video Analysis Backend

Python FastAPI backend for video analysis (scene detection, segmentation, object detection, face recognition, and face-identity fusion).

## Setup

This project uses [UV](https://docs.astral.sh/uv/) for dependency management.
Use Python `3.13.x` for local development and test runs.

```bash
cd backend
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
uv run python run.py
```

## API Endpoints

- `POST /analyze-video` - Upload video, returns `job_id` (HTTP 202)
  - Optional form field: `cleanup_local_video_after_upload` (`true`/`false`) to override local source cleanup policy per request.
- `GET /status/{job_id}` - Poll job status
- `GET /results/{job_id}` - Get analysis JSON with signed R2 file URLs when completed

## Pipeline Stage Order

`process_video()` executes stages in this order:

1. source upload to R2 and source verification
2. scene detection + keyframe extraction
3. per-keyframe CV analysis (segmentation, detection, face, enrichment)
4. video-level tracking outputs:
   - object tracking summary (`video_object_tracks`)
   - optional face identity + person fusion (`video_face_identities`, `video_person_tracks`)
5. artifact verification and job completion

The runtime pipeline stops after CV/tracking outputs. It does not dispatch queue worker tasks and does not build corpus artifacts.

## Cloudflare R2 Configuration

Set the following environment variables before running the API:

- `R2_ACCOUNT_ID` - Cloudflare account identifier
- `R2_BUCKET` - R2 bucket name used for uploads and frame artifacts
- `R2_ACCESS_KEY_ID` - R2 S3 API access key ID
- `R2_SECRET_ACCESS_KEY` - R2 S3 API secret access key
- `R2_URL_TTL_SECONDS` - Signed URL lifetime for `GET /results` file links (default `3600`)
- `R2_RETENTION_DAYS` - Intended retention period for job artifacts (default `7`)
- `R2_ABORT_MULTIPART_DAYS` - Intended incomplete multipart abort window (default `1`)
- `TEMP_MEDIA_DIR` - Local temp directory for transient processing files (default `backend/tmp_media`)
- `CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT` - Default local source cleanup policy after verified upload (`true` by default)

Startup logs include validation for missing required R2 settings.

### Automatic `.env` Loading

At startup, the backend automatically loads dotenv files from the backend directory in this order:

1. `backend/.env`
2. `backend/.env.local`

`backend/.env.local` is the recommended place for machine-local secrets/config.
Environment variables already present in the process take precedence over file values.

## Analysis Artifact Layout (JSON)

Per-frame analysis artifacts are stored in deterministic keys under the same job namespace as source video and frame images:

- `jobs/<job_id>/analysis/json/frame_<N>.json`

This keeps JSON outputs linked to:

- source video: `jobs/<job_id>/input/source.<ext>`
- frame images: `jobs/<job_id>/frames/{original|seg|det|face}/frame_<N>.jpg`

## Local Staging and Cleanup Policy

- Uploads are staged locally first at `TEMP_MEDIA_DIR/<job_id>/input/source.<ext>`.
- Source video is deleted locally only after R2 upload verification succeeds and effective policy allows cleanup.
- Effective cleanup policy is resolved as:
  1. request override `cleanup_local_video_after_upload` (if provided)
  2. fallback to `CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT`
- When cleanup is disabled, the scheduler preserves the retained source video and removes stale non-source artifacts.

## Dependencies

The ML stack is **PyTorch-first** for in-repo pipeline execution and does not require TensorFlow:

- **ultralytics** (YOLO) — object detection and instance segmentation
- **facenet-pytorch** (MTCNN) — face detection
- **scenedetect** — video scene boundary detection
- **opencv-python** — frame I/O and drawing

## Face Identity

Continuous face identity runs on a PyTorch-only EdgeFace path and is enabled by default.

Default behavior (can be set explicitly):

```bash
ENABLE_FACE_IDENTITY_PIPELINE=true
FACE_IDENTITY_MODEL_ID=edgeface_s_gamma_05
FACE_IDENTITY_BACKEND=cpu
FACE_IDENTITY_WEIGHTS_PATH=/absolute/path/to/edgeface_weights.pt
```

Supported profile values:

- `edgeface_s_gamma_05` (default)
- `edgeface_xs_gamma_06` (lower-cost alternative)

Legacy alias `edgeface-arcface-torch` is accepted and normalized to `edgeface_s_gamma_05`.

Optional tuning defaults (already applied when unset):

```bash
FACE_IDENTITY_SCENE_SIMILARITY_THRESHOLD=0.68
FACE_IDENTITY_VIDEO_SIMILARITY_THRESHOLD=0.74
FACE_IDENTITY_AMBIGUITY_MARGIN=0.03
```

Rollback controls:

- Disable face identity with `ENABLE_FACE_IDENTITY_PIPELINE=false`.
- Swap profile via `FACE_IDENTITY_MODEL_ID` and/or weights via `FACE_IDENTITY_WEIGHTS_PATH`.
- Force backend via `FACE_IDENTITY_BACKEND=cuda|mps|cpu`.

## GPU

All models run on PyTorch. CUDA is auto-detected at startup via `torch.cuda.is_available()`.

For GPU acceleration, install the CUDA build of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version (for example `cu118` for CUDA 11.8). Then install the rest:

```bash
pip install -r requirements.txt
```

If CUDA is not available, all models fall back to CPU automatically.
