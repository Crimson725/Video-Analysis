# Local Corpus Retrieval Pipeline

The current corpus pipeline persists retrieval bundles only. Legacy knowledge-graph and embedding build/ingest support is removed.

## Configure environment

Set backend environment values (for example in `backend/.env.local`):

```env
ENABLE_CORPUS_PIPELINE=true
ENABLE_CORPUS_INGEST=false
```

`ENABLE_CORPUS_INGEST` is currently ignored and should remain disabled until the replacement graph/embedding pipeline is introduced.

## Run backend

```bash
cd backend
uv run python run.py
```

Process a video through `/analyze-video`; after scene understanding, the backend will:

1. Build retrieval corpus bundle output.
2. Persist retrieval artifact under `jobs/<job_id>/corpus/rag/...`.

Queue-mode scene-understanding workflow (API + worker in separate terminals):

```bash
# Terminal 1
cd backend
SCENE_AI_EXECUTION_MODE=queue uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Terminal 2
cd backend
SCENE_AI_EXECUTION_MODE=queue uv run python -m app.worker
```

## Troubleshooting

- Corpus output missing: confirm `ENABLE_CORPUS_PIPELINE=true`.
- Unexpected ingest behavior: ensure `ENABLE_CORPUS_INGEST=false` while legacy ingest is removed.
