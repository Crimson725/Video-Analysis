"""Integration-style tests for queue-mode scene AI orchestration and worker behavior."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import jobs
from app.scene_ai_worker import SceneAIWorker
from app.scene_task_queue import QUEUE_STATUS_DEAD_LETTER, QUEUE_STATUS_SUCCEEDED, InMemorySceneTaskQueue, SceneTask

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


def _settings(**overrides):
    defaults = {
        "enable_scene_understanding_pipeline": True,
        "scene_ai_execution_mode": "queue",
        "scene_ai_max_attempts": 2,
        "scene_ai_lease_timeout_seconds": 30,
        "scene_ai_retry_backoff_seconds": 1,
        "scene_ai_retry_backoff_multiplier": 2,
        "scene_ai_retry_backoff_max_seconds": 60,
        "scene_ai_worker_poll_interval_seconds": 1,
        "scene_ai_failure_policy": "fail_job",
        "scene_ai_prompt_version": "v1",
        "scene_ai_runtime_version": "v1",
        "scene_model_id": "gemini-3-flash-preview",
        "synopsis_model_id": "gemini-3-flash-preview",
        "enable_corpus_pipeline": True,
        "enable_corpus_ingest": False,
        "cleanup_local_video_after_upload_default": True,
        "r2_url_ttl_seconds": 3600,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_queue_mode_enforces_stage_order_cv_then_scene_then_corpus():
    queue = InMemorySceneTaskQueue()
    job_id = jobs.create_job()
    order: list[str] = []
    settings = _settings()

    with (
        patch("app.main.SETTINGS", settings),
        patch("app.main.get_scene_task_queue", return_value=queue),
        patch("app.main.get_media_store") as mock_store_factory,
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame") as mock_analyze_frame,
    ):
        mock_store = MagicMock(name="store")
        mock_store.upload_source_video.return_value = f"jobs/{job_id}/input/source.mp4"
        mock_store.verify_object.return_value = True
        mock_store_factory.return_value = mock_store
        mock_model_loader.get.return_value = MagicMock()
        mock_analyze_frame.side_effect = lambda *args, **kwargs: (
            order.append("cv"),
            _frame_payload(job_id),
        )[1]

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")

    task_id = jobs.get_job(job_id)["scene_task_id"]  # type: ignore[index]
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-order",
        media_store_factory=lambda: MagicMock(name="worker_store"),
    )
    with (
        patch(
            "app.scene_ai_worker.video_understanding.run_scene_understanding_pipeline",
            side_effect=lambda **kwargs: (
                order.append("scene_ai"),
                {"scene_narratives": [], "video_synopsis": None},
            )[1],
        ),
        patch(
            "app.scene_ai_worker.corpus.build",
            side_effect=lambda **kwargs: (
                order.append("corpus"),
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
            )[1],
        ),
    ):
        assert worker.process_next_task() is True

    assert order == ["cv", "scene_ai", "corpus"]
    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    task = queue.get_task(task_id=task_id)
    assert task is not None
    assert task.status == QUEUE_STATUS_SUCCEEDED


def test_retry_backoff_and_dead_letter_transitions_include_non_retryable_case():
    queue = InMemorySceneTaskQueue()
    settings = _settings(scene_ai_max_attempts=2, scene_ai_failure_policy="fail_job")
    retry_job = jobs.create_job()
    retry_task = queue.enqueue_task(
        job_id=retry_job,
        payload={
            "job_id": retry_job,
            "scenes": [[0.0, 1.0]],
            "frame_results": [{"frame_id": 0, "timestamp": "00:00:01.000"}],
            "source_key": f"jobs/{retry_job}/input/source.mp4",
        },
        idempotency_key=f"{retry_job}:scene_understanding:v1",
        max_attempts=2,
    )
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-retry",
        media_store_factory=lambda: MagicMock(name="store"),
    )

    with patch(
        "app.scene_ai_worker.video_understanding.run_scene_understanding_pipeline",
        side_effect=RuntimeError("provider timeout"),
    ):
        assert worker.process_next_task() is True
        task_after_first = queue.get_task(task_id=retry_task.task_id)
        assert task_after_first is not None
        assert task_after_first.status == "retry"
        task_after_first.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
        assert worker.process_next_task() is True

    task_after_second = queue.get_task(task_id=retry_task.task_id)
    assert task_after_second is not None
    assert task_after_second.status == QUEUE_STATUS_DEAD_LETTER
    retry_job_state = jobs.get_job(retry_job)
    assert retry_job_state is not None
    assert retry_job_state["status"] == "failed"

    non_retry_job = jobs.create_job()
    non_retry_task = queue.enqueue_task(
        job_id=non_retry_job,
        payload={
            "job_id": non_retry_job,
            "scenes": [[0.0, 1.0]],
            "frame_results": [],
            "source_key": "",
        },
        idempotency_key=f"{non_retry_job}:scene_understanding:v1",
        max_attempts=2,
    )
    assert worker.process_next_task() is True
    non_retry_state = queue.get_task(task_id=non_retry_task.task_id)
    assert non_retry_state is not None
    assert non_retry_state.status == QUEUE_STATUS_DEAD_LETTER


def test_duplicate_delivery_keeps_single_terminal_success():
    queue = InMemorySceneTaskQueue()
    settings = _settings(enable_corpus_pipeline=False)
    job_id = jobs.create_job()
    enqueued = queue.enqueue_task(
        job_id=job_id,
        payload={
            "job_id": job_id,
            "scenes": [[0.0, 1.0]],
            "frame_results": [{"frame_id": 0, "timestamp": "00:00:01.000"}],
            "source_key": f"jobs/{job_id}/input/source.mp4",
        },
        idempotency_key=f"{job_id}:scene_understanding:v1",
        max_attempts=2,
    )
    worker_a = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-a",
        media_store_factory=lambda: MagicMock(name="store"),
    )
    worker_b = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-b",
        media_store_factory=lambda: MagicMock(name="store"),
    )
    with patch(
        "app.scene_ai_worker.video_understanding.run_scene_understanding_pipeline",
        return_value={"scene_narratives": [], "video_synopsis": None},
    ):
        assert worker_a.process_next_task() is True
        succeeded = queue.get_task(task_id=enqueued.task_id)
        assert succeeded is not None
        assert succeeded.status == QUEUE_STATUS_SUCCEEDED
        duplicate = SceneTask(
            task_id=succeeded.task_id,
            job_id=succeeded.job_id,
            task_type=succeeded.task_type,
            status="processing",
            attempts=succeeded.attempts,
            max_attempts=succeeded.max_attempts,
            lease_owner="worker-b",
            lease_expires_at=datetime.now(UTC) + timedelta(seconds=30),
            next_attempt_at=succeeded.next_attempt_at,
            idempotency_key=succeeded.idempotency_key,
            payload=succeeded.payload,
            result_metadata=None,
            last_error=None,
            error_metadata=None,
            created_at=succeeded.created_at,
            updated_at=succeeded.updated_at,
            completed_at=None,
        )
        worker_b._execute_task(duplicate)

    final = queue.get_task(task_id=enqueued.task_id)
    assert final is not None
    assert final.status == QUEUE_STATUS_SUCCEEDED


def test_queue_mode_with_scene_disabled_keeps_empty_scene_outputs():
    queue = InMemorySceneTaskQueue()
    settings = _settings(
        enable_scene_understanding_pipeline=False,
        enable_corpus_pipeline=False,
        enable_corpus_ingest=False,
    )
    job_id = jobs.create_job()

    with (
        patch("app.main.SETTINGS", settings),
        patch("app.main.get_scene_task_queue", return_value=queue),
        patch("app.main.get_media_store") as mock_store_factory,
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.scene.detect_scenes", return_value=[(0.0, 1.0)]),
        patch(
            "app.main.scene.extract_keyframes",
            return_value=[{"frame_id": 0, "timestamp": "00:00:01.000", "image": object()}],
        ),
        patch("app.main.scene.save_original_frames"),
        patch("app.main.analysis.analyze_frame", return_value=_frame_payload(job_id)),
    ):
        mock_store = MagicMock(name="store")
        mock_store.upload_source_video.return_value = f"jobs/{job_id}/input/source.mp4"
        mock_store.verify_object.return_value = True
        mock_store_factory.return_value = mock_store
        mock_model_loader.get.return_value = MagicMock()

        from app.main import process_video

        process_video(job_id, "/tmp/nonexistent.mp4", "mp4")

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["scene_narratives"] == []
    assert job["result"]["video_synopsis"] is None
    assert "scene_task_id" not in job
