"""Tests for retrieval corpus pipeline integration in app.main.process_video."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import jobs


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
            "enrichment": {},
        },
        "analysis_artifacts": {
            "json": f"jobs/{job_id}/analysis/json/frame_0.json",
        },
        "metadata": {
            "provenance": {
                "job_id": job_id,
                "scene_id": None,
                "frame_id": 0,
                "timestamp": "00:00:01.000",
                "source_artifact_key": f"jobs/{job_id}/frames/original/frame_0.jpg",
            },
            "model_provenance": [],
            "evidence_anchors": [],
        },
    }


def test_process_video_runs_corpus_build_when_enabled():
    job_id = jobs.create_job()
    call_order: list[str] = []

    with (
        patch("app.main.get_media_store") as mock_store_factory,
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[
                {"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}
            ],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame") as mock_analyze_frame,
        patch("app.main.corpus.build") as mock_corpus_build,
        patch(
            "app.main.SETTINGS",
            SimpleNamespace(
                enable_corpus_pipeline=True,
                enable_corpus_ingest=True,
                cleanup_local_video_after_upload_default=True,
                r2_url_ttl_seconds=3600,
            ),
        ),
    ):
        mock_store = MagicMock()
        mock_store.upload_source_video.return_value = f"jobs/{job_id}/input/source.mp4"
        mock_store.verify_object.return_value = True
        mock_store_factory.return_value = mock_store

        mock_model_loader.get.return_value = MagicMock()
        mock_analyze_frame.side_effect = lambda *args, **kwargs: (
            call_order.append("cv"),
            _frame_payload(job_id),
        )[1]

        mock_corpus_build.side_effect = lambda **kwargs: (
            call_order.append("corpus"),
            {
                "retrieval": {"job_id": job_id, "chunks": []},
                "artifacts": {
                    "retrieval_bundle": f"jobs/{job_id}/corpus/rag/bundle.json",
                },
            },
        )[1]

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["corpus"] is not None
    assert call_order == ["cv", "corpus"]

    assert mock_corpus_build.call_count == 1
    assert "scene_outputs" not in mock_corpus_build.call_args.kwargs


def test_process_video_corpus_build_receives_frame_results():
    """Corpus build receives frame_results for frame-based chunking."""
    job_id = jobs.create_job()

    with (
        patch("app.main.get_media_store") as mock_store_factory,
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[
                {"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}
            ],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame", return_value=_frame_payload(job_id)),
        patch("app.main.corpus.build") as mock_corpus_build,
        patch(
            "app.main.SETTINGS",
            SimpleNamespace(
                enable_corpus_pipeline=True,
                enable_corpus_ingest=False,
                cleanup_local_video_after_upload_default=True,
                r2_url_ttl_seconds=3600,
            ),
        ),
    ):
        mock_store = MagicMock()
        mock_store.upload_source_video.return_value = f"jobs/{job_id}/input/source.mp4"
        mock_store.verify_object.return_value = True
        mock_store_factory.return_value = mock_store
        mock_model_loader.get.return_value = MagicMock()

        mock_corpus_build.return_value = {
            "retrieval": {"job_id": job_id, "chunks": []},
            "artifacts": {
                "retrieval_bundle": f"jobs/{job_id}/corpus/rag/bundle.json",
            },
        }

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert mock_corpus_build.call_count == 1
    # Corpus build should receive frame_results for frame-based chunking
    assert "frame_results" in mock_corpus_build.call_args.kwargs
    assert len(mock_corpus_build.call_args.kwargs["frame_results"]) == 1
    assert "scene_outputs" not in mock_corpus_build.call_args.kwargs
