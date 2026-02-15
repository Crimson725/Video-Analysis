"""Unit tests for app.main signed URL materialization helpers."""

from app.main import _materialize_signed_result_urls


class _StubMediaStore:
    def sign_read_url(self, key: str, expires_in: int | None = None) -> str:
        return f"https://signed.example/{key}?exp={expires_in}"


def test_materialize_signed_result_urls_normalizes_frame_fields():
    payload = {
        "job_id": "job-123",
        "frames": [
            {
                "frame_id": 7,
                "timestamp": "00:00:07.000",
                "files": {
                    "original": "jobs/job-123/frames/original/frame_7.jpg",
                    "preview": "https://cdn.example/frame_7.jpg",
                },
                "analysis": {
                    "semantic_segmentation": [],
                    "object_detection": [{"label": "person"}, "ignore-me"],
                    "face_recognition": [{"face_id": 1}, "ignore-me"],
                },
                "analysis_artifacts": {
                    "json": "jobs/job-123/analysis/json/frame_7.json",
                },
                "metadata": None,
            }
        ],
        "scene_narratives": [],
        "video_synopsis": None,
        "corpus": None,
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    frame = result["frames"][0]
    assert frame["files"]["original"].startswith("https://signed.example/jobs/")
    assert frame["files"]["preview"] == "https://cdn.example/frame_7.jpg"
    assert frame["analysis"]["object_detection"][0]["track_id"] == "track_7_1"
    assert frame["analysis"]["face_recognition"][0]["identity_id"] == "face_1"
    assert frame["metadata"]["provenance"]["job_id"] == "job-123"
    assert frame["metadata"]["provenance"]["source_artifact_key"].startswith(
        "https://signed.example/jobs/"
    )


def test_materialize_signed_result_urls_signs_nested_artifacts():
    payload = {
        "job_id": "job-8",
        "frames": [],
        "scene_narratives": [
            {
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 2.0,
                "narrative_paragraph": "A short summary.",
                "key_moments": [],
                "artifacts": {
                    "packet": "jobs/job-8/scene/packets/scene_0.json",
                    "narrative": "jobs/job-8/scene/narratives/scene_0.json",
                },
                "corpus": {
                    "artifacts": {
                        "graph_bundle": "jobs/job-8/corpus/graph/scene_0.json",
                        "retrieval_bundle": "jobs/job-8/corpus/rag/scene_0.json",
                    }
                },
            }
        ],
        "video_synopsis": {
            "synopsis": "full summary",
            "artifact": "jobs/job-8/summary/synopsis.json",
            "model": "gemini",
        },
        "corpus": {
            "artifacts": {
                "retrieval_bundle": "jobs/job-8/corpus/rag/bundle.json",
            }
        },
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    scene = result["scene_narratives"][0]
    assert scene["artifacts"]["packet"].startswith("https://signed.example/jobs/")
    assert scene["artifacts"]["narrative"].startswith("https://signed.example/jobs/")
    assert scene["corpus"]["artifacts"]["graph_bundle"].startswith("https://signed.example/jobs/")
    assert scene["corpus"]["artifacts"]["retrieval_bundle"].startswith("https://signed.example/jobs/")
    assert result["video_synopsis"]["artifact"].startswith("https://signed.example/jobs/")
    assert result["corpus"]["artifacts"]["retrieval_bundle"].startswith("https://signed.example/jobs/")


def test_materialize_signed_result_urls_defaults_scene_fields_when_missing():
    payload = {
        "job_id": "job-9",
        "frames": [],
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    assert result["frames"] == []
    assert result["scene_narratives"] == []
    assert result["video_synopsis"] is None
    assert result["corpus"] is None


def test_materialize_signed_result_urls_preserves_video_face_identity_summary():
    payload = {
        "job_id": "job-11",
        "frames": [],
        "video_face_identities": {
            "video_identities": [
                {"video_person_id": "video_person_1", "scene_person_ids": ["scene_0_person_1"]}
            ]
        },
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    assert result["video_face_identities"] == payload["video_face_identities"]


def test_materialize_signed_result_urls_preserves_video_person_tracks():
    payload = {
        "job_id": "job-12",
        "frames": [],
        "video_person_tracks": {
            "method": "object_face_fusion_v1",
            "tracks": [{"person_track_id": "person_track_abc"}],
        },
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    assert result["video_person_tracks"] == payload["video_person_tracks"]


def test_materialize_signed_result_urls_ignores_invalid_scene_and_frame_items():
    payload = {
        "job_id": "job-10",
        "frames": ["invalid-frame"],
        "scene_narratives": ["invalid-scene"],
        "video_synopsis": "invalid",
        "corpus": "invalid",
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    assert result["frames"] == []
    assert result["scene_narratives"] == []
    assert result["video_synopsis"] is None
    assert result["corpus"] is None
