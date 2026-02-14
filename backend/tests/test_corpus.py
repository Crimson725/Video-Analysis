"""Tests for retrieval chunk construction in app.corpus."""

from app.corpus import _build_retrieval_bundle


def test_build_retrieval_bundle_prefers_scene_retrieval_chunks():
    bundle = _build_retrieval_bundle(
        job_id="job-1",
        frame_results=[],
        scene_outputs={
            "scene_narratives": [
                {
                    "scene_id": 3,
                    "start_sec": 1.25,
                    "end_sec": 5.0,
                    "narrative_paragraph": "should be ignored when chunks exist",
                    "artifacts": {"narrative": "jobs/job-1/scenes/scene_3.md"},
                    "corpus": {
                        "retrieval_chunks": [
                            {
                                "chunk_id": "scene-3-chunk-1",
                                "text": "  A person opens the door. ",
                                "artifact_keys": ["jobs/job-1/scenes/scene_3.md"],
                                "source_entity_ids": ["person_1"],
                            },
                            {"text": "   "},
                        ]
                    },
                }
            ]
        },
    )

    assert [chunk.chunk_id for chunk in bundle.chunks] == ["scene-3-chunk-1"]
    assert bundle.chunks[0].text == "A person opens the door."
    assert bundle.chunks[0].metadata == {
        "job_id": "job-1",
        "scene_id": 3,
        "start_sec": 1.25,
        "end_sec": 5.0,
        "artifact_keys": ["jobs/job-1/scenes/scene_3.md"],
        "source_entity_ids": ["person_1"],
    }


def test_build_retrieval_bundle_does_not_use_narrative_fallback_when_scene_chunks_present():
    bundle = _build_retrieval_bundle(
        job_id="job-2",
        frame_results=[],
        scene_outputs={
            "scene_narratives": [
                {
                    "scene_id": 9,
                    "narrative_paragraph": "fallback should not be emitted",
                    "corpus": {"retrieval_chunks": [{"text": "   "}]},
                }
            ]
        },
    )

    assert bundle.chunks == []


def test_build_retrieval_bundle_uses_scene_narrative_when_scene_chunks_missing():
    bundle = _build_retrieval_bundle(
        job_id="job-3",
        frame_results=[],
        scene_outputs={
            "scene_narratives": [
                {
                    "scene_id": 5,
                    "start_sec": 2.0,
                    "end_sec": 8.5,
                    "narrative_paragraph": "A crowd gathers near the stage.",
                    "artifacts": {"narrative": "jobs/job-3/scenes/scene_5.md"},
                    "corpus": {},
                }
            ]
        },
    )

    assert len(bundle.chunks) == 1
    chunk = bundle.chunks[0]
    assert chunk.text == "A crowd gathers near the stage."
    assert chunk.metadata == {
        "job_id": "job-3",
        "scene_id": 5,
        "start_sec": 2.0,
        "end_sec": 8.5,
        "artifact_keys": ["jobs/job-3/scenes/scene_5.md"],
    }


def test_build_retrieval_bundle_falls_back_to_frame_descriptions_when_no_scene_chunks():
    bundle = _build_retrieval_bundle(
        job_id="job-4",
        frame_results=[
            {
                "frame_id": 11,
                "timestamp": "00:00:03.500",
                "analysis": {
                    "object_detection": [{"label": "car"}, {"label": "person"}],
                    "face_recognition": [{"identity_id": "face_1"}],
                },
                "analysis_artifacts": {"json": "jobs/job-4/analysis/json/frame_11.json"},
            }
        ],
        scene_outputs={"scene_narratives": []},
    )

    assert len(bundle.chunks) == 1
    chunk = bundle.chunks[0]
    assert chunk.text == "Frame 11 at 00:00:03.500 includes objects [car, person] and faces [face_1]."
    assert chunk.metadata == {
        "job_id": "job-4",
        "frame_id": 11,
        "timestamp": "00:00:03.500",
        "artifact_keys": ["jobs/job-4/analysis/json/frame_11.json"],
    }
