"""Unit tests for queued scene AI worker behavior."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import jobs
from app.scene_ai_worker import SceneAIWorker
from app.scene_ai_worker_contracts import SceneWorkerTaskInput
from app.scene_task_queue import (
    QUEUE_STATUS_DEAD_LETTER,
    QUEUE_STATUS_RETRY,
    QUEUE_STATUS_SUCCEEDED,
    InMemorySceneTaskQueue,
)


def _settings(**overrides):
    defaults = {
        "kg_pipeline_enabled": True,
        "enable_corpus_pipeline": False,
        "enable_corpus_ingest": False,
        "scene_ai_lease_timeout_seconds": 30,
        "scene_ai_retry_backoff_seconds": 1,
        "scene_ai_retry_backoff_multiplier": 2,
        "scene_ai_retry_backoff_max_seconds": 60,
        "scene_ai_worker_poll_interval_seconds": 1,
        "scene_ai_failure_policy": "fail_job",
        "scene_model_id": "gemini-3-flash-preview",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _task_payload(job_id: str, *, video_object_tracks: dict | None = None) -> dict:
    return SceneWorkerTaskInput(
        job_id=job_id,
        scenes=[(0.0, 1.0)],
        frame_results=[{"frame_id": 0, "timestamp": "00:00:01.000"}],
        source_key=f"jobs/{job_id}/input/source.mp4",
        video_object_tracks=video_object_tracks,
    ).to_payload()


def test_worker_completes_task_with_kg_enrichment():
    queue = InMemorySceneTaskQueue()
    settings = _settings()
    job_id = jobs.create_job()
    enqueued = queue.enqueue_task(
        job_id=job_id,
        payload=_task_payload(job_id),
        idempotency_key=f"{job_id}:scene_worker:v1",
        max_attempts=3,
    )
    media_store = MagicMock(name="media_store")
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-test",
        media_store_factory=lambda: media_store,
    )

    fake_kg_result = {
        "scene_id": 0,
        "scene_graph_delta": {"entities": [], "relations": [], "events": []},
        "validation_errors": [],
        "retry_count": 0,
    }

    with (
        patch(
            "app.scene_ai_worker.run_kg_workflow",
            return_value=fake_kg_result,
        ),
        patch.object(SceneAIWorker, "_get_neo4j_writer", return_value=None),
    ):
        processed = worker.process_next_task()

    assert processed is True
    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    result = job["result"]
    assert result["kg_results"] == [fake_kg_result]
    assert result["corpus"] is None
    assert "scene_narratives" not in result
    assert "video_synopsis" not in result
    task = queue.get_task(task_id=enqueued.task_id)
    assert task is not None
    assert task.status == QUEUE_STATUS_SUCCEEDED


def test_worker_completes_with_kg_disabled():
    queue = InMemorySceneTaskQueue()
    settings = _settings(kg_pipeline_enabled=False)
    job_id = jobs.create_job()
    enqueued = queue.enqueue_task(
        job_id=job_id,
        payload=_task_payload(job_id),
        idempotency_key=f"{job_id}:scene_worker:v1",
        max_attempts=1,
    )
    media_store = MagicMock(name="media_store")
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-no-kg",
        media_store_factory=lambda: media_store,
    )

    processed = worker.process_next_task()

    assert processed is True
    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    result = job["result"]
    assert result["kg_results"] == []
    assert result["corpus"] is None
    task = queue.get_task(task_id=enqueued.task_id)
    assert task is not None
    assert task.status == QUEUE_STATUS_SUCCEEDED


def test_worker_payload_round_trip_preserves_video_object_tracks():
    queue = InMemorySceneTaskQueue()
    settings = _settings(kg_pipeline_enabled=False)
    job_id = jobs.create_job()
    object_tracks = {
        "enabled": True,
        "method": "object_tracking_v1",
        "tracks": [{"object_track_id": "object_track_1"}],
    }
    queue.enqueue_task(
        job_id=job_id,
        payload=_task_payload(job_id, video_object_tracks=object_tracks),
        idempotency_key=f"{job_id}:scene_worker:v1",
        max_attempts=1,
    )
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-track-passthrough",
        media_store_factory=lambda: MagicMock(name="media_store"),
    )

    processed = worker.process_next_task()

    assert processed is True
    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["video_object_tracks"] == object_tracks


def test_worker_retries_then_succeeds_after_transient_failure():
    queue = InMemorySceneTaskQueue()
    settings = _settings()
    job_id = jobs.create_job()
    enqueued = queue.enqueue_task(
        job_id=job_id,
        payload=_task_payload(job_id),
        idempotency_key=f"{job_id}:scene_worker:v1",
        max_attempts=3,
    )
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-retry",
        media_store_factory=lambda: MagicMock(name="media_store"),
    )

    fake_kg_result = {
        "scene_id": 0,
        "scene_graph_delta": {"entities": [], "relations": [], "events": []},
        "validation_errors": [],
        "retry_count": 0,
    }

    # First attempt: _get_neo4j_writer raises, causing a retryable error.
    # Second attempt: succeeds normally.
    with (
        patch(
            "app.scene_ai_worker.run_kg_workflow",
            return_value=fake_kg_result,
        ),
        patch.object(
            SceneAIWorker,
            "_get_neo4j_writer",
            side_effect=[RuntimeError("temporary provider outage"), None],
        ),
    ):
        first = worker.process_next_task()
        assert first is True
        task_after_first = queue.get_task(task_id=enqueued.task_id)
        assert task_after_first is not None
        assert task_after_first.status == QUEUE_STATUS_RETRY
        task_after_first.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)

        second = worker.process_next_task()
        assert second is True

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    final_task = queue.get_task(task_id=enqueued.task_id)
    assert final_task is not None
    assert final_task.status == QUEUE_STATUS_SUCCEEDED
    assert final_task.attempts == 2


def test_worker_marks_dead_letter_on_non_retryable_payload_error():
    queue = InMemorySceneTaskQueue()
    settings = _settings(scene_ai_failure_policy="fail_job")
    job_id = jobs.create_job()
    enqueued = queue.enqueue_task(
        job_id=job_id,
        payload={
            "job_id": job_id,
            "scenes": [[0.0, 1.0]],
            "frame_results": [],
            "source_key": "",
        },
        idempotency_key=f"{job_id}:scene_worker:v1",
        max_attempts=3,
    )
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-error",
        media_store_factory=lambda: MagicMock(name="media_store"),
    )

    processed = worker.process_next_task()
    assert processed is True

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "failed"
    task = queue.get_task(task_id=enqueued.task_id)
    assert task is not None
    assert task.status == QUEUE_STATUS_DEAD_LETTER


def test_worker_fallback_policy_completes_with_empty_kg_results():
    queue = InMemorySceneTaskQueue()
    settings = _settings(scene_ai_failure_policy="fallback_empty")
    job_id = jobs.create_job()
    enqueued = queue.enqueue_task(
        job_id=job_id,
        payload={
            "job_id": job_id,
            "scenes": [[0.0, 1.0]],
            "frame_results": [{"frame_id": 0, "timestamp": "00:00:01.000"}],
            "source_key": "",
        },
        idempotency_key=f"{job_id}:scene_worker:v1",
        max_attempts=1,
    )
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-fallback",
        media_store_factory=lambda: MagicMock(name="media_store"),
    )

    processed = worker.process_next_task()
    assert processed is True

    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    result = job["result"]
    assert result["kg_results"] == []
    assert result["corpus"] is None
    assert "scene_narratives" not in result
    assert "video_synopsis" not in result
    task = queue.get_task(task_id=enqueued.task_id)
    assert task is not None
    assert task.status == QUEUE_STATUS_DEAD_LETTER
