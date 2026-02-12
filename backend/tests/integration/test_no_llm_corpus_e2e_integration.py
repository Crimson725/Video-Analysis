"""No-LLM end-to-end integration test for CV -> corpus -> DB ingest -> cleanup."""

from __future__ import annotations

from dataclasses import replace
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from app import jobs
from app.cleanup import RETAIN_SOURCE_MARKER
from app.config import Settings
from app.main import process_video
from app.storage import (
    build_analysis_key,
    build_corpus_key,
    build_frame_key,
    build_scene_key,
    build_source_video_key,
    build_summary_key,
)

pytestmark = pytest.mark.integration


class InMemoryMediaStore:
    """In-memory MediaStore substitute for integration tests without R2."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_source_video(
        self,
        job_id: str,
        file_path: str,
        content_type: str,
        source_extension: str | None = None,
    ) -> str:
        del content_type
        object_key = build_source_video_key(job_id, source_extension=source_extension)
        self.objects[object_key] = Path(file_path).read_bytes()
        return object_key

    def upload_frame_image(self, job_id: str, frame_kind: str, frame_id: int, image_bytes: bytes) -> str:
        object_key = build_frame_key(job_id, frame_kind, frame_id)
        self.objects[object_key] = image_bytes
        return object_key

    def upload_analysis_artifact(self, job_id: str, artifact_kind: str, frame_id: int, payload: bytes) -> str:
        object_key = build_analysis_key(job_id, artifact_kind, frame_id)
        self.objects[object_key] = payload
        return object_key

    def upload_scene_artifact(self, job_id: str, artifact_kind: str, scene_id: int, payload: bytes) -> str:
        object_key = build_scene_key(job_id, artifact_kind, scene_id)
        self.objects[object_key] = payload
        return object_key

    def upload_summary_artifact(self, job_id: str, artifact_kind: str, payload: bytes) -> str:
        object_key = build_summary_key(job_id, artifact_kind)
        self.objects[object_key] = payload
        return object_key

    def upload_corpus_artifact(
        self,
        job_id: str,
        artifact_kind: str,
        payload: bytes,
        filename: str = "bundle.json",
    ) -> str:
        object_key = build_corpus_key(job_id, artifact_kind, filename=filename)
        self.objects[object_key] = payload
        return object_key

    def read_object(self, object_key: str) -> bytes:
        return self.objects[object_key]

    def delete_object(self, object_key: str) -> None:
        self.objects.pop(object_key, None)

    def verify_object(self, object_key: str) -> bool:
        return object_key in self.objects

    def sign_read_url(self, object_key: str, expires_in: int | None = None) -> str:
        del expires_in
        return f"https://signed.example/{object_key}"


def test_process_video_no_llm_corpus_e2e(
    corpus_e2e_real_video_path: str,
    tmp_path: Path,
    corpus_e2e_backend_probe: dict[str, str],
    corpus_e2e_job_id: str,
    corpus_e2e_db_cleanup,
):
    from neo4j import GraphDatabase
    import psycopg

    jobs.jobs.clear()
    source_dir = tmp_path / corpus_e2e_job_id / "input"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "source.mp4"
    shutil.copy2(corpus_e2e_real_video_path, source_path)

    media_store = InMemoryMediaStore()
    base_settings = Settings.from_env()
    test_settings = replace(
        base_settings,
        enable_scene_understanding_pipeline=False,
        enable_corpus_pipeline=True,
        enable_corpus_ingest=True,
        cleanup_local_video_after_upload_default=False,
        graph_backend="neo4j",
        vector_backend="pgvector",
        neo4j_uri=corpus_e2e_backend_probe["neo4j_uri"],
        neo4j_username=corpus_e2e_backend_probe["neo4j_username"],
        neo4j_password=corpus_e2e_backend_probe["neo4j_password"],
        neo4j_database=corpus_e2e_backend_probe["neo4j_database"],
        pgvector_dsn=corpus_e2e_backend_probe["pgvector_dsn"],
    )

    with (
        patch("app.main.SETTINGS", test_settings),
        patch("app.main.TEMP_MEDIA_DIR", tmp_path),
        patch("app.main.get_media_store", return_value=media_store),
    ):
        process_video(
            corpus_e2e_job_id,
            str(source_path),
            "mp4",
            "video/mp4",
            cleanup_local_video_after_upload=False,
        )

    job = jobs.get_job(corpus_e2e_job_id)
    assert job is not None
    assert job["status"] == "completed", f"Unexpected failure: {job.get('error')}"

    payload = job["result"]
    assert payload.get("scene_narratives") == []

    corpus_payload = payload.get("corpus")
    assert isinstance(corpus_payload, dict)
    artifacts = corpus_payload["artifacts"]
    graph_key = artifacts["graph_bundle"]
    retrieval_key = artifacts["retrieval_bundle"]
    embeddings_key = artifacts["embeddings_bundle"]
    assert graph_key.startswith(f"jobs/{corpus_e2e_job_id}/corpus/graph/")
    assert retrieval_key.startswith(f"jobs/{corpus_e2e_job_id}/corpus/rag/")
    assert embeddings_key.startswith(f"jobs/{corpus_e2e_job_id}/corpus/embeddings/")

    graph_bundle = json.loads(media_store.read_object(graph_key).decode("utf-8"))
    retrieval_bundle = json.loads(media_store.read_object(retrieval_key).decode("utf-8"))

    node_ids = [item["node_id"] for item in graph_bundle.get("nodes", [])]
    claim_ids = [item["claim_id"] for item in graph_bundle.get("derived_claims", [])]
    chunk_ids = [item["chunk_id"] for item in retrieval_bundle.get("chunks", [])]

    assert node_ids, "Expected at least one graph node in corpus bundle"
    assert chunk_ids, "Expected at least one retrieval chunk in corpus bundle"

    corpus_e2e_db_cleanup(node_ids=node_ids, claim_ids=claim_ids, chunk_ids=chunk_ids)

    neo4j_driver = GraphDatabase.driver(
        corpus_e2e_backend_probe["neo4j_uri"],
        auth=(
            corpus_e2e_backend_probe["neo4j_username"],
            corpus_e2e_backend_probe["neo4j_password"],
        ),
    )
    try:
        with neo4j_driver.session(database=corpus_e2e_backend_probe["neo4j_database"]) as session:
            node_count = session.run(
                "MATCH (n:CorpusNode) WHERE n.node_id IN $node_ids RETURN count(n) AS count",
                node_ids=node_ids,
            ).single()["count"]
    finally:
        neo4j_driver.close()
    assert node_count >= 1

    with psycopg.connect(corpus_e2e_backend_probe["pgvector_dsn"]) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM corpus_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
            chunk_count = cur.fetchone()[0]
    assert chunk_count >= 1

    retain_marker = tmp_path / corpus_e2e_job_id / RETAIN_SOURCE_MARKER
    assert source_path.exists(), "Source video should be retained when cleanup is disabled"
    assert retain_marker.exists(), "Retention marker must exist when cleanup is disabled"
