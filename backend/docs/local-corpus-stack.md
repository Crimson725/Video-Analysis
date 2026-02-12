# Local KG/RAG Stack (Neo4j + pgvector)

This change adds a local-first corpus pipeline that can build and ingest graph + retrieval artifacts without managed cloud dependencies.

## 1. Start local services

```bash
docker compose -f backend/docker-compose.corpus.yml up -d
```

You can use Podman with the same compose file:

```bash
podman compose -f backend/docker-compose.corpus.yml up -d
```

## 2. Configure environment

Set the backend environment values (for example in `backend/.env.local`):

```env
ENABLE_CORPUS_PIPELINE=true
ENABLE_CORPUS_INGEST=true
GRAPH_BACKEND=neo4j
VECTOR_BACKEND=pgvector
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=local-dev-password
NEO4J_DATABASE=neo4j
PGVECTOR_DSN=postgresql://video_analysis:video_analysis@127.0.0.1:5433/video_analysis
EMBEDDING_MODEL_ID=local-hash-embedding
EMBEDDING_MODEL_VERSION=v1
EMBEDDING_DIMENSION=16
```

## 3. Run backend

```bash
cd backend
uv run python run.py
```

Process a video through `/analyze-video`; after scene understanding, the backend will:

1. Build corpus graph/retrieval/embeddings bundles.
2. Persist artifacts under `jobs/<job_id>/corpus/...`.
3. Optionally ingest into Neo4j + pgvector when `ENABLE_CORPUS_INGEST=true`.

## 4. Quick verification

Neo4j:

```cypher
MATCH (n:CorpusNode) RETURN count(n);
```

PostgreSQL:

```sql
SELECT count(*) FROM corpus_chunks;
```

## 5. Low-cost deployment path

The same adapters can run on low-cost/self-hosted targets:

- Neo4j Community on a small VM/container host.
- PostgreSQL + pgvector on low-cost Postgres-compatible providers or self-hosted Postgres.

No changes to corpus builder contracts are required when swapping only adapter configuration.

## 6. No-LLM Corpus E2E Integration Test

This test runs real `process_video` CV processing with:

- scene understanding disabled (`ENABLE_SCENE_UNDERSTANDING_PIPELINE=false`)
- corpus build/ingest enabled
- local Neo4j + pgvector assertions
- cleanup policy verification
- canonical real input video from `Test Videos/WhatCarCanYouGetForAGrand.mp4` (no synthetic MP4 fixtures)
- isolated default DB ports (`bolt://127.0.0.1:17687`, `postgresql://...@127.0.0.1:15433/...`) to avoid conflicts with developer-local DB services

Run with Docker:

```bash
cd backend
RUN_CORPUS_E2E_INTEGRATION=1 \
ENABLE_SCENE_UNDERSTANDING_PIPELINE=false \
ENABLE_CORPUS_PIPELINE=true \
ENABLE_CORPUS_INGEST=true \
uv run pytest tests/integration/test_no_llm_corpus_e2e_integration.py -m integration -vv
```

Run with Podman:

```bash
cd backend
RUN_CORPUS_E2E_INTEGRATION=1 \
ENABLE_SCENE_UNDERSTANDING_PIPELINE=false \
ENABLE_CORPUS_PIPELINE=true \
ENABLE_CORPUS_INGEST=true \
uv run pytest tests/integration/test_no_llm_corpus_e2e_integration.py -m integration -vv
```

The test is intentionally opt-in. It is skipped unless `RUN_CORPUS_E2E_INTEGRATION=1`.
When opt-in is enabled, fixture setup attempts to auto-start local Neo4j + pgvector using
`podman compose` first (with Docker fallback) before declaring the backends unavailable.
You can override the default test DB targets by setting `NEO4J_URI` and `PGVECTOR_DSN`.

## 7. Troubleshooting

- Neo4j not reachable: verify `podman compose -f backend/docker-compose.corpus.yml ps` (or Docker equivalent) and confirm `bolt://127.0.0.1:7687`.
- pgvector not reachable: verify Postgres is listening on `127.0.0.1:5433` and `PGVECTOR_DSN` credentials are correct.
- pgvector auth error like `role "video_analysis" does not exist`: reset the test-stack volumes and recreate:
  `podman compose -p video-analysis-corpus-itest -f backend/docker-compose.corpus.yml down -v --remove-orphans && podman compose -p video-analysis-corpus-itest -f backend/docker-compose.corpus.yml up -d` (Docker equivalent if using Docker).
- Missing DB client libs: run `cd backend && uv sync` to install `neo4j` and `psycopg[binary]`.
- Test skipped unexpectedly: confirm `RUN_CORPUS_E2E_INTEGRATION=1` is set in the same shell invocation.
