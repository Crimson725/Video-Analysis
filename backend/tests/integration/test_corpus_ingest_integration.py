"""Integration tests for corpus ingest into Neo4j + pgvector."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app import corpus
from app.corpus_ingest import build_graph_adapter, build_vector_adapter, ingest_corpus

pytestmark = [pytest.mark.integration, pytest.mark.external_api]


def _settings(gemini_api_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_dimension=16,
        embedding_model_id="gemini-embedding-001",
        embedding_model_version="v1",
        google_api_key=gemini_api_key,
        graph_backend="neo4j",
        vector_backend="pgvector",
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:47687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "local-dev-password"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        pgvector_dsn=os.getenv(
            "PGVECTOR_DSN",
            "postgresql://video_analysis:video_analysis@127.0.0.1:45433/video_analysis",
        ),
    )


def _prereqs_available() -> bool:
    try:
        import neo4j  # noqa: F401
        import psycopg  # noqa: F401
    except Exception:
        return False
    return True


def _frame(job_id: str, frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": "00:00:01.000",
        "files": {
            "original": f"jobs/{job_id}/frames/original/frame_{frame_id}.jpg",
            "segmentation": f"jobs/{job_id}/frames/seg/frame_{frame_id}.jpg",
            "detection": f"jobs/{job_id}/frames/det/frame_{frame_id}.jpg",
            "face": f"jobs/{job_id}/frames/face/frame_{frame_id}.jpg",
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
            "json": f"jobs/{job_id}/analysis/json/frame_{frame_id}.json",
            "toon": f"jobs/{job_id}/analysis/toon/frame_{frame_id}.toon",
        },
        "metadata": {
            "provenance": {
                "job_id": job_id,
                "scene_id": None,
                "frame_id": frame_id,
                "timestamp": "00:00:01.000",
                "source_artifact_key": f"jobs/{job_id}/frames/original/frame_{frame_id}.jpg",
            },
            "model_provenance": [],
            "evidence_anchors": [],
        },
    }


def _wait_for_backends(settings: SimpleNamespace, timeout_seconds: int = 90) -> None:
    from neo4j import GraphDatabase
    import psycopg

    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
            try:
                with driver.session(database=settings.neo4j_database) as session:
                    session.run("RETURN 1").single()
            finally:
                driver.close()

            with psycopg.connect(settings.pgvector_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return
        except Exception as exc:
            last_error = str(exc)
            lowered = last_error.lower()
            if "role \"" in lowered and "does not exist" in lowered:
                pytest.skip(
                    "PostgreSQL role is missing for pgvector integration. "
                    "Recreate local DB volumes, then rerun integration tests. "
                    f"Detail: {last_error}"
                )
            time.sleep(2)

    pytest.skip(
        "Neo4j/pgvector backends are not reachable for corpus ingest integration. "
        f"Last probe error: {last_error}"
    )


def _cleanup_pgvector_rows(pgvector_dsn: str, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return

    import psycopg

    try:
        with psycopg.connect(pgvector_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM corpus_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
            conn.commit()
    except Exception:
        pass


def _cleanup_neo4j_rows(
    *,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    neo4j_database: str,
    node_ids: list[str],
    claim_ids: list[str],
) -> None:
    if not node_ids and not claim_ids:
        return

    from neo4j import GraphDatabase

    driver = None
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        with driver.session(database=neo4j_database) as session:
            if node_ids:
                session.run(
                    "MATCH (n:CorpusNode) WHERE n.node_id IN $node_ids DETACH DELETE n",
                    node_ids=node_ids,
                )
            if claim_ids:
                session.run(
                    "MATCH (c:CorpusClaim) WHERE c.claim_id IN $claim_ids DELETE c",
                    claim_ids=claim_ids,
                )
    except Exception:
        pass
    finally:
        if driver is not None:
            driver.close()


@pytest.mark.skipif(not _prereqs_available(), reason="Missing Neo4j/psycopg client dependencies")
def test_ingest_into_local_neo4j_and_pgvector(gemini_api_key):
    settings = _settings(gemini_api_key)
    _wait_for_backends(settings)

    job_id = f"job-int-{uuid4().hex[:12]}"
    scene_outputs = {
        "scene_narratives": [
            {
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 3.0,
                "narrative_paragraph": "A person appears in the scene.",
                "key_moments": ["person appears"],
                "artifacts": {
                    "packet": f"jobs/{job_id}/scene/packets/scene_0.toon",
                    "narrative": f"jobs/{job_id}/scene/narratives/scene_0.json",
                },
                "corpus": {
                    "entities": [],
                    "events": [],
                    "relations": [],
                    "retrieval_chunks": [],
                    "artifacts": {
                        "graph_bundle": f"jobs/{job_id}/corpus/graph/scene_0.json",
                        "retrieval_bundle": f"jobs/{job_id}/corpus/rag/scene_0.json",
                    },
                },
            }
        ],
        "video_synopsis": None,
    }

    corpus_payload = corpus.build(
        job_id=job_id,
        scenes=[(0.0, 3.0)],
        frame_results=[_frame(job_id, 0)],
        scene_outputs=scene_outputs,
        settings=settings,
        media_store=None,
    )

    node_ids = [item["node_id"] for item in corpus_payload["graph"]["nodes"]]
    claim_ids = [item["claim_id"] for item in corpus_payload["graph"]["derived_claims"]]
    chunk_ids = [item["chunk_id"] for item in corpus_payload["retrieval"]["chunks"]]

    graph_adapter = build_graph_adapter(settings)
    vector_adapter = build_vector_adapter(settings)

    try:
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
        try:
            with driver.session(database=settings.neo4j_database) as session:
                node_count = session.run(
                    "MATCH (n:CorpusNode) WHERE n.node_id IN $node_ids RETURN count(n) AS count",
                    node_ids=node_ids,
                ).single()["count"]
        finally:
            driver.close()
        assert node_count >= 1

        with psycopg.connect(settings.pgvector_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM corpus_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
                chunk_count = cur.fetchone()[0]
        assert chunk_count >= 1
    finally:
        _cleanup_pgvector_rows(settings.pgvector_dsn, chunk_ids)
        _cleanup_neo4j_rows(
            neo4j_uri=settings.neo4j_uri,
            neo4j_username=settings.neo4j_username,
            neo4j_password=settings.neo4j_password,
            neo4j_database=settings.neo4j_database,
            node_ids=node_ids,
            claim_ids=claim_ids,
        )
        close_method = getattr(graph_adapter, "close", None)
        if callable(close_method):
            close_method()
