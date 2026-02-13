"""Unit tests for corpus ingest adapters."""

from app import corpus
from app.corpus_ingest import InMemoryGraphAdapter, InMemoryVectorAdapter, ingest_corpus


class _Settings:
    embedding_dimension = 8
    embedding_model_id = "gemini-embedding-001"
    embedding_model_version = "v1"
    google_api_key = "test-key"


class _StubEmbeddingClient:
    def embed_documents(
        self,
        texts: list[str],
        *,
        output_dimensionality: int | None = None,
    ) -> list[list[float]]:
        assert output_dimensionality is not None
        return [[0.25] * output_dimensionality for _ in texts]


def _payload() -> dict:
    frame_results = [
        {
            "frame_id": 0,
            "timestamp": "00:00:01.000",
            "files": {
                "original": "jobs/job-1/frames/original/frame_0.jpg",
                "segmentation": "jobs/job-1/frames/seg/frame_0.jpg",
                "detection": "jobs/job-1/frames/det/frame_0.jpg",
                "face": "jobs/job-1/frames/face/frame_0.jpg",
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
                "face_recognition": [],
                "enrichment": {},
            },
            "analysis_artifacts": {
                "json": "jobs/job-1/analysis/json/frame_0.json",
                "toon": "jobs/job-1/analysis/toon/frame_0.toon",
            },
            "metadata": {
                "provenance": {
                    "job_id": "job-1",
                    "scene_id": None,
                    "frame_id": 0,
                    "timestamp": "00:00:01.000",
                    "source_artifact_key": "jobs/job-1/frames/original/frame_0.jpg",
                },
                "model_provenance": [],
                "evidence_anchors": [],
            },
        }
    ]
    scene_outputs = {
        "scene_narratives": [
            {
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 3.0,
                "narrative_paragraph": "A person appears in the scene.",
                "key_moments": ["person appears"],
                "artifacts": {
                    "packet": "jobs/job-1/scene/packets/scene_0.toon",
                    "narrative": "jobs/job-1/scene/narratives/scene_0.json",
                },
                "corpus": {
                    "entities": [],
                    "events": [],
                    "relations": [],
                    "retrieval_chunks": [],
                    "artifacts": {
                        "graph_bundle": "jobs/job-1/corpus/graph/scene_0.json",
                        "retrieval_bundle": "jobs/job-1/corpus/rag/scene_0.json",
                    },
                },
            }
        ],
        "video_synopsis": None,
    }
    return corpus.build(
        job_id="job-1",
        scenes=[(0.0, 3.0)],
        frame_results=frame_results,
        scene_outputs=scene_outputs,
        settings=_Settings(),
        embedding_client=_StubEmbeddingClient(),
    )


def test_in_memory_ingest_is_idempotent():
    payload = _payload()
    graph = InMemoryGraphAdapter()
    vector = InMemoryVectorAdapter()

    first = ingest_corpus(corpus_payload=payload, graph_adapter=graph, vector_adapter=vector)
    second = ingest_corpus(corpus_payload=payload, graph_adapter=graph, vector_adapter=vector)

    assert first["graph"]["nodes"] >= 1
    assert first["vector"]["chunks"] >= 1
    assert second["graph"]["nodes"] == first["graph"]["nodes"]
    assert second["vector"]["chunks"] == first["vector"]["chunks"]
    chunk_payload = next(iter(vector.chunks.values()))
    assert chunk_payload["model_id"] == "gemini-embedding-001"
    assert chunk_payload["model_version"] == "v1"
    assert len(chunk_payload["embedding"]) == _Settings.embedding_dimension
