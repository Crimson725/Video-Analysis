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
                    "toon": "jobs/job-123/analysis/toon/frame_7.toon",
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
                    "packet": "jobs/job-8/scene/packets/scene_0.toon",
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
                "graph_bundle": "jobs/job-8/corpus/graph/bundle.json",
                "retrieval_bundle": "jobs/job-8/corpus/rag/bundle.json",
                "embeddings_bundle": "jobs/job-8/corpus/embeddings/bundle.json",
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
    assert result["corpus"]["artifacts"]["graph_bundle"].startswith("https://signed.example/jobs/")
    assert result["corpus"]["artifacts"]["retrieval_bundle"].startswith("https://signed.example/jobs/")
    assert result["corpus"]["artifacts"]["embeddings_bundle"].startswith("https://signed.example/jobs/")
