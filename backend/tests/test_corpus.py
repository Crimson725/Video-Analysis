"""Tests for retrieval chunk construction in app.corpus."""

from app.corpus import _build_retrieval_bundle


def test_build_retrieval_bundle_creates_frame_chunks():
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
                "analysis_artifacts": {
                    "json": "jobs/job-4/analysis/json/frame_11.json"
                },
            }
        ],
    )

    assert len(bundle.chunks) == 1
    chunk = bundle.chunks[0]
    assert (
        chunk.text
        == "Frame 11 at 00:00:03.500 includes objects [car, person] and faces [face_1]."
    )
    assert chunk.metadata == {
        "job_id": "job-4",
        "frame_id": 11,
        "timestamp": "00:00:03.500",
        "artifact_keys": ["jobs/job-4/analysis/json/frame_11.json"],
    }


def test_build_retrieval_bundle_empty_frames():
    bundle = _build_retrieval_bundle(
        job_id="job-5",
        frame_results=[],
    )
    assert bundle.chunks == []


def test_build_retrieval_bundle_deduplicates_by_chunk_id():
    frame = {
        "frame_id": 1,
        "timestamp": "00:00:01.000",
        "analysis": {
            "object_detection": [{"label": "dog"}],
            "face_recognition": [],
        },
        "analysis_artifacts": {"json": "jobs/job-6/analysis/json/frame_1.json"},
    }
    bundle = _build_retrieval_bundle(
        job_id="job-6",
        frame_results=[frame],
    )
    assert len(bundle.chunks) == 1
