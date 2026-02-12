"""Unit tests for corpus construction and deterministic IDs."""

from types import SimpleNamespace

from app import corpus


def _settings(**overrides):
    defaults = {
        "embedding_dimension": 8,
        "embedding_model_id": "local-hash-embedding",
        "embedding_model_version": "v1",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _frame(frame_id: int, timestamp: str) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "files": {
            "original": f"jobs/job-1/frames/original/frame_{frame_id}.jpg",
            "segmentation": f"jobs/job-1/frames/seg/frame_{frame_id}.jpg",
            "detection": f"jobs/job-1/frames/det/frame_{frame_id}.jpg",
            "face": f"jobs/job-1/frames/face/frame_{frame_id}.jpg",
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
            "enrichment": {
                "ocr_blocks": [
                    {
                        "text": "EXIT",
                        "confidence": 0.8,
                        "bbox": [1, 1, 6, 3],
                    }
                ]
            },
        },
        "analysis_artifacts": {
            "json": f"jobs/job-1/analysis/json/frame_{frame_id}.json",
            "toon": f"jobs/job-1/analysis/toon/frame_{frame_id}.toon",
        },
        "metadata": {
            "provenance": {
                "job_id": "job-1",
                "scene_id": None,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "source_artifact_key": f"jobs/job-1/frames/original/frame_{frame_id}.jpg",
            },
            "model_provenance": [],
            "evidence_anchors": [],
        },
    }


def _scene_outputs() -> dict:
    return {
        "scene_narratives": [
            {
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 4.0,
                "narrative_paragraph": "A person appears in frame and looks at EXIT sign.",
                "key_moments": ["person appears"],
                "artifacts": {
                    "packet": "jobs/job-1/scene/packets/scene_0.toon",
                    "narrative": "jobs/job-1/scene/narratives/scene_0.json",
                },
                "corpus": {
                    "entities": [
                        {
                            "entity_id": "entity_person",
                            "label": "person",
                            "entity_type": "object",
                            "count": 1,
                            "confidence": 0.9,
                            "temporal_span": {
                                "first_seen": 0.0,
                                "last_seen": 1.0,
                                "duration_sec": 1.0,
                            },
                            "evidence": [
                                {
                                    "frame_id": 0,
                                    "timestamp": "00:00:01.000",
                                    "artifact_key": "jobs/job-1/analysis/json/frame_0.json",
                                    "bbox": [0, 0, 10, 10],
                                    "text_span": "person",
                                }
                            ],
                            "track_id": "person_track_1",
                            "identity_id": None,
                        }
                    ],
                    "events": [],
                    "relations": [],
                    "retrieval_chunks": [
                        {
                            "chunk_id": "chunk_1",
                            "text": "A person appears near EXIT sign.",
                            "source_entity_ids": ["entity_person"],
                            "artifact_keys": ["jobs/job-1/scene/packets/scene_0.toon"],
                            "temporal_span": {
                                "first_seen": 0.0,
                                "last_seen": 4.0,
                                "duration_sec": 4.0,
                            },
                        }
                    ],
                    "artifacts": {
                        "graph_bundle": "jobs/job-1/corpus/graph/scene_0.json",
                        "retrieval_bundle": "jobs/job-1/corpus/rag/scene_0.json",
                    },
                },
            }
        ],
        "video_synopsis": None,
    }


def test_corpus_ids_are_deterministic():
    frame_results = [_frame(0, "00:00:01.000")]
    scene_outputs = _scene_outputs()

    first = corpus.build(
        job_id="job-1",
        scenes=[(0.0, 4.0)],
        frame_results=frame_results,
        scene_outputs=scene_outputs,
        settings=_settings(),
    )
    second = corpus.build(
        job_id="job-1",
        scenes=[(0.0, 4.0)],
        frame_results=frame_results,
        scene_outputs=scene_outputs,
        settings=_settings(),
    )

    first_node_ids = sorted(item["node_id"] for item in first["graph"]["nodes"])
    second_node_ids = sorted(item["node_id"] for item in second["graph"]["nodes"])
    assert first_node_ids == second_node_ids

    first_chunk_ids = sorted(item["chunk_id"] for item in first["retrieval"]["chunks"])
    second_chunk_ids = sorted(item["chunk_id"] for item in second["retrieval"]["chunks"])
    assert first_chunk_ids == second_chunk_ids


def test_source_facts_and_derived_claims_are_separate_and_grounded():
    payload = corpus.build(
        job_id="job-1",
        scenes=[(0.0, 4.0)],
        frame_results=[_frame(0, "00:00:01.000")],
        scene_outputs=_scene_outputs(),
        settings=_settings(),
    )

    assert payload["graph"]["source_facts"], "expected source facts"
    assert payload["graph"]["derived_claims"], "expected derived claims"

    for fact in payload["graph"]["source_facts"]:
        assert fact["evidence"], "source fact must include evidence"
        assert "frame_id" in fact["evidence"][0]
        assert "artifact_key" in fact["evidence"][0]

    for claim in payload["graph"]["derived_claims"]:
        assert claim["evidence"], "derived claim must include evidence"
        assert claim["provenance"]["source"] == "scene_narrative"
