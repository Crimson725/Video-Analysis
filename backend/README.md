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
- `GET /status/{job_id}` - Poll job status
- `GET /results/{job_id}` - Get analysis JSON when completed
- `GET /static/{path}` - Serve generated images

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
