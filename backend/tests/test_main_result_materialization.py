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
                "raw_frame_index": 175,
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
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    frame = result["frames"][0]
    assert result["pipeline"]["stages"] == []
    assert frame["files"]["original"].startswith("https://signed.example/jobs/")
    assert frame["files"]["preview"] == "https://cdn.example/frame_7.jpg"
    assert frame["analysis"]["object_detection"][0]["track_id"] == "track_7_1"
    assert frame["analysis"]["face_recognition"][0]["identity_id"] == "face_1"
    assert frame["raw_frame_index"] == 175
    assert frame["metadata"]["provenance"]["raw_frame_index"] == 175
    assert frame["metadata"]["provenance"]["job_id"] == "job-123"
    assert frame["metadata"]["provenance"]["source_artifact_key"].startswith(
        "https://signed.example/jobs/"
    )


def test_materialize_signed_result_urls_ignores_non_cv_fields():
    payload = {
        "job_id": "job-8",
        "frames": [],
        "corpus": {"artifacts": {"retrieval_bundle": "jobs/job-8/corpus/rag/bundle.json"}},
        "video_face_identities": {"video_identities": []},
        "video_person_tracks": {"tracks": []},
        "video_object_tracks": {"tracks": []},
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    assert result["job_id"] == "job-8"
    assert result["frames"] == []
    assert result["pipeline"]["stages"] == []


def test_materialize_signed_result_urls_defaults_when_missing():
    payload = {
        "job_id": "job-9",
        "frames": [],
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    assert result["frames"] == []
    assert result["pipeline"]["stages"] == []


def test_materialize_signed_result_urls_ignores_invalid_frame_items():
    payload = {
        "job_id": "job-10",
        "frames": ["invalid-frame"],
        "corpus": "invalid",
    }

    result = _materialize_signed_result_urls(payload, _StubMediaStore())

    assert result["frames"] == []
