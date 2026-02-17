# Video Analysis

Video Analysis is a FastAPI-based backend project for analyzing uploaded videos.  
It processes scenes and keyframes, runs computer vision analysis, and returns job-based results via API endpoints.

## Project Structure

- `backend/` - main FastAPI service and pipeline code
- `backend/tests/` - unit and integration tests
- `backend/docs/` - deeper backend and local stack documentation
- `Test Videos/` - sample videos for local testing

## Prerequisites

- Python `3.13.x`
- [uv](https://docs.astral.sh/uv/) for dependency management and running commands

## Setup

```bash
cd backend
uv sync
```

Create `backend/.env.local` for machine-local secrets/config as needed (for example `GOOGLE_API_KEY`, R2 settings).

## Run Locally

From `backend/`:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Alternative:

```bash
uv run python run.py
```

## Basic API Endpoints

- `POST /analyze-video` - upload video and start analysis (returns `job_id`)
- `GET /status/{job_id}` - check processing status
- `GET /results/{job_id}` - retrieve completed results

## Tests

From the repository root:

```bash
make test-unit
```

```bash
make test-integration
```

## More Details

- Backend details: `backend/README.md`
- Pipeline overview: `video_analysis_pipeline.md`
