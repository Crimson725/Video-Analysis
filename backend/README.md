# Video Analysis Backend

Python FastAPI backend for video analysis (scene detection, segmentation, object detection, face recognition).

## Setup

This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

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

## Pipeline Stage Order (LLM Boundary)

`process_video()` executes stages in this order:

1. scene detection + keyframe extraction
2. per-keyframe CV analysis (segmentation, detection, face, enrichment)
3. scene-understanding generation (`scene_narratives`, `video_synopsis`) when `ENABLE_SCENE_UNDERSTANDING_PIPELINE=true`
4. corpus/KG/retrieval/embeddings build when `ENABLE_CORPUS_PIPELINE=true`

LLM involvement is constrained to stage 3 (scene understanding). When scene understanding is disabled, results keep a stable shape with:

- `scene_narratives: []`
- `video_synopsis: null`

## Corpus/KG/RAG Pipeline Flags

These flags are enabled by default and can be overridden per environment:

- `ENABLE_SCENE_UNDERSTANDING_PIPELINE` - build scene packets, scene narratives, and video synopsis (`true` default)
- `ENABLE_CORPUS_PIPELINE` - build graph/retrieval/embeddings bundles after scene understanding (`true` default)
- `ENABLE_CORPUS_INGEST` - ingest bundles into configured graph/vector adapters (`true` default)
- `GRAPH_BACKEND` - `neo4j` or `memory`
- `VECTOR_BACKEND` - `pgvector` or `memory`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `PGVECTOR_DSN`
- `EMBEDDING_MODEL_ID` (default `gemini-embedding-001`)
- `EMBEDDING_MODEL_VERSION` (default `v1`)
- `EMBEDDING_DIMENSION` (default `16`)

When using Gemini embeddings (default), `GOOGLE_API_KEY` is required for corpus embedding generation.

Local stack setup guide: `/Users/crimson2049/Video Analysis/backend/docs/local-corpus-stack.md`

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

`backend/.env.local` is the recommended place for machine-local secrets such as `GOOGLE_API_KEY`.
Environment variables already present in the process take precedence over file values.

## R2 Lifecycle Operations

Use Cloudflare R2 lifecycle rules in each environment to:

- Abort incomplete multipart uploads after the configured threshold (`R2_ABORT_MULTIPART_DAYS`)
- Expire old job artifact objects under the `jobs/` prefix according to retention policy (`R2_RETENTION_DAYS`)

Reference helper payload: `/Users/crimson2049/Video Analysis/backend/docs/r2-lifecycle.example.json`

## Analysis Artifact Layout (JSON + TOON)

Per-frame analysis artifacts are stored in deterministic keys under the same job namespace as source video and frame images:

- `jobs/<job_id>/analysis/json/frame_<N>.json`
- `jobs/<job_id>/analysis/toon/frame_<N>.toon`

This keeps JSON/TOON outputs linked to:

- source video: `jobs/<job_id>/input/source.<ext>`
- frame images: `jobs/<job_id>/frames/{original|seg|det|face}/frame_<N>.jpg`

## Local Staging and Cleanup Policy

- Uploads are staged locally first at `TEMP_MEDIA_DIR/<job_id>/input/source.<ext>`.
- Source video is deleted locally only after R2 upload verification succeeds and effective policy allows cleanup.
- Effective cleanup policy is resolved as:
  1. request override `cleanup_local_video_after_upload` (if provided)
  2. fallback to `CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT`
- When cleanup is disabled, the scheduler preserves the retained source video and removes stale non-source artifacts.

## TOON Conversion Runtime

JSON-to-TOON conversion uses `@toon-format/toon` through a small Node helper runtime.

Install helper dependencies once:

```bash
cd backend/scripts/toon_runtime
npm install
```

The backend invokes `backend/scripts/toon_runtime/convert_toon.mjs` during frame artifact persistence.

## R2 JSON/TOON Integration Test

Run the targeted R2 read/write test:

```bash
cd backend
uv run pytest tests/integration/test_r2_analysis_artifacts_integration.py -m integration -v
```

Behavior:

- Skips when required R2 credentials/config are missing.
- Writes JSON and TOON artifacts to R2, reads them back, and validates non-empty payloads.
- Always deletes test-created objects in teardown/finally, including failure-path scenarios.

## No-LLM Corpus E2E Integration Test

This integration test validates one full path: real CV `process_video` run, corpus artifact generation, Neo4j + pgvector ingest verification, and cleanup policy assertion with scene/synopsis LLM generation disabled.

```bash
cd backend
ENABLE_SCENE_UNDERSTANDING_PIPELINE=false \
ENABLE_CORPUS_PIPELINE=true \
ENABLE_CORPUS_INGEST=true \
GOOGLE_API_KEY="<your-gemini-key>" \
uv run pytest tests/integration/test_no_llm_corpus_e2e_integration.py -m integration -vv
```

Notes:

- Works with either Docker or Podman as long as local Neo4j + pgvector services are reachable.
- Corpus embedding generation uses Gemini embeddings by default and requires `GOOGLE_API_KEY`.
- The integration fixture attempts to start local Neo4j + pgvector automatically (`podman compose` first, Docker fallback) before skipping.
- The suite uses isolated default DB endpoints (`bolt://127.0.0.1:47687`, `postgresql://video_analysis:video_analysis@127.0.0.1:45433/video_analysis`) to avoid conflicts with existing local DB services.
- The suite is pinned to the canonical real test video at `/Users/crimson2049/Video Analysis/Test Videos/WhatCarCanYouGetForAGrand.mp4` and does not use fake MP4 fixtures.
- Default `pytest` runs remain unchanged (`-m 'not integration'`), but `-m integration` runs now include this test.

## Video Synopsis End-to-End Integration Test (Live Gemini)

Run the targeted synopsis E2E test:

```bash
cd backend
GOOGLE_API_KEY="<your-gemini-key>" ENABLE_SCENE_UNDERSTANDING_PIPELINE=true \
  uv run pytest tests/integration/test_video_synopsis_e2e_integration.py -m "integration and external_api" -vv
```

Behavior:

- Skips when `GOOGLE_API_KEY` is missing.
- Skips when Gemini client setup is unavailable.
- Skips when Gemini quota/rate-limit availability prevents a live run.
- Uses live `process_video()` execution and validates `scene_narratives` plus `video_synopsis` contract fields.
- Keeps assertions structural (no exact generated-text matching).

## Output-Content E2E Integration Test (Canonical Test Video)

Run the dedicated output-content validation suite for the canonical test video:

```bash
cd backend
GOOGLE_API_KEY="<your-gemini-key>" ENABLE_SCENE_UNDERSTANDING_PIPELINE=true \
  uv run pytest tests/integration/test_output_content_e2e_integration.py -m "integration and external_api" -vv
```

Behavior:

- Uses `Test Videos/WhatCarCanYouGetForAGrand.mp4` as the canonical fixture input.
- Verifies completion contract plus output-content rubric checks (minimum narrative/synopsis usefulness and topical anchor coverage).
- Avoids brittle exact-text assertions and model-accuracy metrics.
- Skips with actionable prerequisite guidance when required env/config is unavailable.

Tuning knobs:

- Edit `backend/tests/integration/fixtures/output_content_expectations.json` to adjust topic anchors and minimum thresholds without changing test assertion logic.

## Dependencies

The ML stack is **PyTorch-only** — no TensorFlow or ONNX Runtime:

- **ultralytics** (YOLO) — object detection and instance segmentation
- **facenet-pytorch** (MTCNN) — face detection
- **scenedetect** — video scene boundary detection
- **opencv-python** — frame I/O and drawing

## GPU

All models run on PyTorch. CUDA is auto-detected at startup via `torch.cuda.is_available()`.

For GPU acceleration, install the CUDA build of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version (e.g., `cu118` for CUDA 11.8). Then install the rest:

```bash
pip install -r requirements.txt
```

If CUDA is not available, all models fall back to CPU automatically.
