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
