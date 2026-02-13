"""Integration-style test for scene summary outputs in process_video."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import jobs
from app.main import process_video


pytestmark = pytest.mark.integration


def _frame_payload(job_id: str) -> dict:
    return {
        "frame_id": 0,
        "timestamp": "00:00:01.000",
        "files": {
            "original": f"jobs/{job_id}/frames/original/frame_0.jpg",
            "segmentation": f"jobs/{job_id}/frames/seg/frame_0.jpg",
            "detection": f"jobs/{job_id}/frames/det/frame_0.jpg",
            "face": f"jobs/{job_id}/frames/face/frame_0.jpg",
        },
        "analysis": {
            "semantic_segmentation": [],
            "object_detection": [],
            "face_recognition": [],
        },
        "analysis_artifacts": {
            "json": f"jobs/{job_id}/analysis/json/frame_0.json",
            "toon": f"jobs/{job_id}/analysis/toon/frame_0.toon",
        },
    }


def test_process_video_includes_scene_outputs(tmp_path):
    video_path = Path(tmp_path) / "source.mp4"
    video_path.write_bytes(b"video")

    fake_store = MagicMock()
    fake_store.upload_source_video.return_value = "jobs/job-1/input/source.mp4"
    fake_store.verify_object.return_value = True

    frame_payload = {
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
            "object_detection": [],
            "face_recognition": [],
        },
        "analysis_artifacts": {
            "json": "jobs/job-1/analysis/json/frame_0.json",
            "toon": "jobs/job-1/analysis/toon/frame_0.toon",
        },
    }

    scene_outputs = {
        "scene_narratives": [
            {
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "narrative_paragraph": "Scene summary.",
                "key_moments": ["moment 1"],
                "artifacts": {
                    "packet": "jobs/job-1/scene/packets/scene_0.toon",
                    "narrative": "jobs/job-1/scene/narratives/scene_0.json",
                },
            }
        ],
        "video_synopsis": {
            "synopsis": "Video synopsis.",
            "artifact": "jobs/job-1/summary/synopsis.json",
            "model": "gemini-2.5-flash-lite",
        },
    }

    settings = SimpleNamespace(
        cleanup_local_video_after_upload_default=True,
        enable_scene_understanding_pipeline=True,
        enable_corpus_pipeline=False,
        enable_corpus_ingest=False,
    )

    job_id = jobs.create_job()
    with (
        patch("app.main.SETTINGS", settings),
        patch("app.main.TEMP_MEDIA_DIR", Path(tmp_path) / "tmp_media"),
        patch("app.main.get_media_store", return_value=fake_store),
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame", return_value=frame_payload),
        patch(
            "app.main.video_understanding.run_scene_understanding_pipeline",
            return_value=scene_outputs,
        ),
        patch("app.main.ModelLoader") as mock_model_loader,
    ):
        mock_model_loader.get.return_value = MagicMock()
        process_video(job_id, str(video_path), "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert len(job["result"]["scene_narratives"]) == 1
    assert job["result"]["video_synopsis"]["artifact"] == "jobs/job-1/summary/synopsis.json"


def test_process_video_runs_scene_understanding_between_cv_and_corpus(tmp_path):
    video_path = Path(tmp_path) / "source.mp4"
    video_path.write_bytes(b"video")

    fake_store = MagicMock()
    fake_store.upload_source_video.return_value = "jobs/job-2/input/source.mp4"
    fake_store.verify_object.return_value = True

    job_id = jobs.create_job()
    call_order: list[str] = []

    settings = SimpleNamespace(
        cleanup_local_video_after_upload_default=True,
        enable_scene_understanding_pipeline=True,
        enable_corpus_pipeline=True,
        enable_corpus_ingest=False,
    )

    scene_outputs = {
        "scene_narratives": [
            {
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "narrative_paragraph": "Scene summary.",
                "key_moments": ["moment 1"],
                "artifacts": {
                    "packet": f"jobs/{job_id}/scene/packets/scene_0.toon",
                    "narrative": f"jobs/{job_id}/scene/narratives/scene_0.json",
                },
            }
        ],
        "video_synopsis": {
            "synopsis": "Video synopsis.",
            "artifact": f"jobs/{job_id}/summary/synopsis.json",
            "model": "gemini-2.5-flash-lite",
        },
    }

    corpus_payload = {
        "graph": {"job_id": job_id, "nodes": [], "edges": [], "source_facts": [], "derived_claims": []},
        "retrieval": {"job_id": job_id, "chunks": []},
        "embeddings": {"job_id": job_id, "dimension": 8, "embeddings": []},
        "artifacts": {
            "graph_bundle": f"jobs/{job_id}/corpus/graph/bundle.json",
            "retrieval_bundle": f"jobs/{job_id}/corpus/rag/bundle.json",
            "embeddings_bundle": f"jobs/{job_id}/corpus/embeddings/bundle.json",
        },
    }

    with (
        patch("app.main.SETTINGS", settings),
        patch("app.main.TEMP_MEDIA_DIR", Path(tmp_path) / "tmp_media"),
        patch("app.main.get_media_store", return_value=fake_store),
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame") as mock_analyze_frame,
        patch("app.main.video_understanding.run_scene_understanding_pipeline") as mock_scene_pipeline,
        patch("app.main.corpus.build") as mock_corpus_build,
        patch("app.main.ModelLoader") as mock_model_loader,
    ):
        mock_model_loader.get.return_value = MagicMock()

        def _analyze_side_effect(*args, **kwargs):
            call_order.append("cv")
            return _frame_payload(job_id)

        def _scene_side_effect(**kwargs):
            call_order.append("scene_llm")
            return scene_outputs

        def _corpus_side_effect(**kwargs):
            call_order.append("corpus")
            return corpus_payload

        mock_analyze_frame.side_effect = _analyze_side_effect
        mock_scene_pipeline.side_effect = _scene_side_effect
        mock_corpus_build.side_effect = _corpus_side_effect

        process_video(job_id, str(video_path), "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert call_order == ["cv", "scene_llm", "corpus"]
    assert job["result"]["scene_narratives"]
    assert job["result"]["video_synopsis"] is not None


def test_process_video_skips_scene_understanding_and_keeps_stable_scene_shape(tmp_path):
    video_path = Path(tmp_path) / "source.mp4"
    video_path.write_bytes(b"video")

    fake_store = MagicMock()
    fake_store.upload_source_video.return_value = "jobs/job-3/input/source.mp4"
    fake_store.verify_object.return_value = True

    job_id = jobs.create_job()

    settings = SimpleNamespace(
        cleanup_local_video_after_upload_default=True,
        enable_scene_understanding_pipeline=False,
        enable_corpus_pipeline=True,
        enable_corpus_ingest=False,
    )

    captured_scene_outputs: dict[str, object] = {}
    corpus_payload = {
        "graph": {"job_id": job_id, "nodes": [], "edges": [], "source_facts": [], "derived_claims": []},
        "retrieval": {"job_id": job_id, "chunks": []},
        "embeddings": {"job_id": job_id, "dimension": 8, "embeddings": []},
        "artifacts": {
            "graph_bundle": f"jobs/{job_id}/corpus/graph/bundle.json",
            "retrieval_bundle": f"jobs/{job_id}/corpus/rag/bundle.json",
            "embeddings_bundle": f"jobs/{job_id}/corpus/embeddings/bundle.json",
        },
    }

    with (
        patch("app.main.SETTINGS", settings),
        patch("app.main.TEMP_MEDIA_DIR", Path(tmp_path) / "tmp_media"),
        patch("app.main.get_media_store", return_value=fake_store),
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame", return_value=_frame_payload(job_id)),
        patch("app.main.video_understanding.run_scene_understanding_pipeline") as mock_scene_pipeline,
        patch("app.main.corpus.build") as mock_corpus_build,
        patch("app.main.ModelLoader") as mock_model_loader,
    ):
        mock_model_loader.get.return_value = MagicMock()

        def _corpus_side_effect(**kwargs):
            captured_scene_outputs.update(kwargs["scene_outputs"])
            return corpus_payload

        mock_corpus_build.side_effect = _corpus_side_effect
        process_video(job_id, str(video_path), "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["scene_narratives"] == []
    assert job["result"]["video_synopsis"] is None
    mock_scene_pipeline.assert_not_called()
    assert captured_scene_outputs == {
        "scene_narratives": [],
        "video_synopsis": None,
    }
