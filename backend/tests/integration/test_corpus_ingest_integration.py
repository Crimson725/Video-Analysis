"""Integration tests for corpus ingest into Neo4j + pgvector."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from app import corpus
from app.corpus_ingest import build_graph_adapter, build_vector_adapter, ingest_corpus

pytestmark = [pytest.mark.integration, pytest.mark.external_api]


def _settings(gemini_api_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_dimension=8,
        embedding_model_id="gemini-embedding-001",
        embedding_model_version="v1",
        google_api_key=gemini_api_key,
        graph_backend="neo4j",
        vector_backend="pgvector",
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "local-dev-password"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        pgvector_dsn=os.getenv(
            "PGVECTOR_DSN",
            "postgresql://video_analysis:video_analysis@127.0.0.1:5433/video_analysis",
        ),
    )


def _prereqs_available() -> bool:
    try:
        import neo4j  # noqa: F401
        import psycopg  # noqa: F401
    except Exception:
        return False
    return bool(os.getenv("RUN_CORPUS_INGEST_INTEGRATION") == "1")


def _frame(frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": "00:00:01.000",
        "files": {
            "original": f"jobs/job-int/frames/original/frame_{frame_id}.jpg",
            "segmentation": f"jobs/job-int/frames/seg/frame_{frame_id}.jpg",
            "detection": f"jobs/job-int/frames/det/frame_{frame_id}.jpg",
            "face": f"jobs/job-int/frames/face/frame_{frame_id}.jpg",
        },
        "analysis": {
            "semantic_segmentation": [],
            "object_detection": [
                {
                    "track_id": "person_track_1",
                    "label": "person",
                    "confidence": 0.9,
                    "box": [0, 0, 10, 10],
                }
            ],
            "face_recognition": [
                {
                    "face_id": 1,
                    "identity_id": "face_1",
                    "confidence": 0.95,
                    "coordinates": [2, 2, 8, 8],
                }
            ],
            "enrichment": {},
        },
        "analysis_artifacts": {
            "json": f"jobs/job-int/analysis/json/frame_{frame_id}.json",
            "toon": f"jobs/job-int/analysis/toon/frame_{frame_id}.toon",
        },
        "metadata": {
            "provenance": {
                "job_id": "job-int",
                "scene_id": None,
                "frame_id": frame_id,
                "timestamp": "00:00:01.000",
                "source_artifact_key": f"jobs/job-int/frames/original/frame_{frame_id}.jpg",
            },
            "model_provenance": [],
            "evidence_anchors": [],
        },
    }


@pytest.mark.skipif(not _prereqs_available(), reason="Set RUN_CORPUS_INGEST_INTEGRATION=1 with Neo4j/pgvector running")
def test_ingest_into_local_neo4j_and_pgvector(gemini_api_key):
    settings = _settings(gemini_api_key)
    scene_outputs = {
        "scene_narratives": [
            {
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 3.0,
                "narrative_paragraph": "A person appears in the scene.",
                "key_moments": ["person appears"],
                "artifacts": {
                    "packet": "jobs/job-int/scene/packets/scene_0.toon",
                    "narrative": "jobs/job-int/scene/narratives/scene_0.json",
                },
                "corpus": {
                    "entities": [],
                    "events": [],
                    "relations": [],
                    "retrieval_chunks": [],
                    "artifacts": {
                        "graph_bundle": "jobs/job-int/corpus/graph/scene_0.json",
                        "retrieval_bundle": "jobs/job-int/corpus/rag/scene_0.json",
                    },
                },
            }
        ],
        "video_synopsis": None,
    }

    corpus_payload = corpus.build(
        job_id="job-int",
        scenes=[(0.0, 3.0)],
        frame_results=[_frame(0)],
        scene_outputs=scene_outputs,
        settings=settings,
        media_store=None,
    )

    graph_adapter = build_graph_adapter(settings)
    vector_adapter = build_vector_adapter(settings)
    report = ingest_corpus(
        corpus_payload=corpus_payload,
        graph_adapter=graph_adapter,
        vector_adapter=vector_adapter,
    )

    assert report["graph"]["nodes"] >= 1
    assert report["vector"]["chunks"] >= 1

    from neo4j import GraphDatabase
    import psycopg

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    with driver.session(database=settings.neo4j_database) as session:
        node_count = session.run(
            "MATCH (n:CorpusNode) RETURN count(n) AS count"
        ).single()["count"]
    assert node_count >= 1

    with psycopg.connect(settings.pgvector_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM corpus_chunks")
            chunk_count = cur.fetchone()[0]
    assert chunk_count >= 1
