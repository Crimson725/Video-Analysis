"""Tests for corpus pipeline integration in app.main.process_video."""

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
            "toon": f"jobs/{job_id}/analysis/toon/frame_0.toon",
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


def test_process_video_runs_corpus_build_and_ingest_when_enabled():
    job_id = jobs.create_job()
    call_order: list[str] = []

    with (
        patch("app.main.get_media_store") as mock_store_factory,
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame") as mock_analyze_frame,
        patch("app.main.video_understanding.run_scene_understanding_pipeline") as mock_scene_pipeline,
        patch("app.main.corpus.build") as mock_corpus_build,
        patch("app.main.corpus_ingest.build_graph_adapter") as mock_graph_adapter_builder,
        patch("app.main.corpus_ingest.build_vector_adapter") as mock_vector_adapter_builder,
        patch("app.main.corpus_ingest.ingest_corpus") as mock_ingest,
        patch(
            "app.main.SETTINGS",
            SimpleNamespace(
                enable_scene_understanding_pipeline=True,
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

        mock_scene_pipeline.side_effect = lambda **kwargs: (
            call_order.append("scene_llm"),
            {
                "scene_narratives": [],
                "video_synopsis": None,
            },
        )[1]
        mock_corpus_build.side_effect = lambda **kwargs: (
            call_order.append("corpus"),
            {
                "graph": {"job_id": job_id, "nodes": [], "edges": [], "source_facts": [], "derived_claims": []},
                "retrieval": {"job_id": job_id, "chunks": []},
                "embeddings": {"job_id": job_id, "dimension": 8, "embeddings": []},
                "artifacts": {
                    "graph_bundle": f"jobs/{job_id}/corpus/graph/bundle.json",
                    "retrieval_bundle": f"jobs/{job_id}/corpus/rag/bundle.json",
                    "embeddings_bundle": f"jobs/{job_id}/corpus/embeddings/bundle.json",
                },
            },
        )[1]

        mock_scene_outputs = {
            "scene_narratives": [],
            "video_synopsis": None,
        }
        mock_scene_pipeline.return_value = mock_scene_outputs

        mock_graph_adapter_builder.return_value = MagicMock()
        mock_vector_adapter_builder.return_value = MagicMock()
        mock_ingest.return_value = {"graph": {"nodes": 0}, "vector": {"chunks": 0}}

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["corpus"] is not None
    assert job["result"]["corpus"]["ingest"] == {"graph": {"nodes": 0}, "vector": {"chunks": 0}}
    assert call_order == ["cv", "scene_llm", "corpus"]

    assert mock_corpus_build.call_count == 1
    assert mock_corpus_build.call_args.kwargs["scene_outputs"] == {
        "scene_narratives": [],
        "video_synopsis": None,
    }


def test_process_video_passes_default_scene_outputs_to_corpus_when_scene_stage_disabled():
    job_id = jobs.create_job()

    with (
        patch("app.main.get_media_store") as mock_store_factory,
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame", return_value=_frame_payload(job_id)),
        patch("app.main.video_understanding.run_scene_understanding_pipeline") as mock_scene_pipeline,
        patch("app.main.corpus.build") as mock_corpus_build,
        patch(
            "app.main.SETTINGS",
            SimpleNamespace(
                enable_scene_understanding_pipeline=False,
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
            "graph": {"job_id": job_id, "nodes": [], "edges": [], "source_facts": [], "derived_claims": []},
            "retrieval": {"job_id": job_id, "chunks": []},
            "embeddings": {"job_id": job_id, "dimension": 8, "embeddings": []},
            "artifacts": {
                "graph_bundle": f"jobs/{job_id}/corpus/graph/bundle.json",
                "retrieval_bundle": f"jobs/{job_id}/corpus/rag/bundle.json",
                "embeddings_bundle": f"jobs/{job_id}/corpus/embeddings/bundle.json",
            },
        }

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["scene_narratives"] == []
    assert job["result"]["video_synopsis"] is None
    mock_scene_pipeline.assert_not_called()
    assert mock_corpus_build.call_count == 1
    assert mock_corpus_build.call_args.kwargs["scene_outputs"] == {
        "scene_narratives": [],
        "video_synopsis": None,
    }
