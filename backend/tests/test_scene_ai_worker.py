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
        "enable_scene_understanding_pipeline": True,
        "enable_corpus_pipeline": False,
        "enable_corpus_ingest": False,
        "scene_ai_lease_timeout_seconds": 30,
        "scene_ai_retry_backoff_seconds": 1,
        "scene_ai_retry_backoff_multiplier": 2,
        "scene_ai_retry_backoff_max_seconds": 60,
        "scene_ai_worker_poll_interval_seconds": 1,
        "scene_ai_failure_policy": "fail_job",
        "scene_ai_prompt_version": "v1",
        "scene_ai_runtime_version": "v1",
        "scene_model_id": "gemini-2.5-flash-lite",
        "synopsis_model_id": "gemini-2.5-flash-lite",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _task_payload(job_id: str) -> dict:
    return SceneWorkerTaskInput(
        job_id=job_id,
        scenes=[(0.0, 1.0)],
        frame_results=[{"frame_id": 0, "timestamp": "00:00:01.000"}],
        source_key=f"jobs/{job_id}/input/source.mp4",
    ).to_payload()


def test_worker_completes_task_and_persists_provenance():
    queue = InMemorySceneTaskQueue()
    settings = _settings()
    job_id = jobs.create_job()
    enqueued = queue.enqueue_task(
        job_id=job_id,
        payload=_task_payload(job_id),
        idempotency_key=f"{job_id}:scene_understanding:v1",
        max_attempts=3,
    )
    media_store = MagicMock(name="media_store")
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-test",
        media_store_factory=lambda: media_store,
    )

    with patch(
        "app.scene_ai_worker.video_understanding.run_scene_understanding_pipeline",
        return_value={
            "scene_narratives": [
                {
                    "scene_id": 0,
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "narrative_paragraph": "Scene summary.",
                    "key_moments": ["moment"],
                    "artifacts": {
                        "packet": f"jobs/{job_id}/scene/packets/scene_0.toon",
                        "narrative": f"jobs/{job_id}/scene/narratives/scene_0.json",
                    },
                    "trace": {"stage": "scene_narrative"},
                }
            ],
            "video_synopsis": {
                "synopsis": "Video summary.",
                "artifact": f"jobs/{job_id}/summary/synopsis.json",
                "model": "gemini-2.5-flash-lite",
                "trace": {"stage": "video_synopsis"},
            },
        },
    ):
        processed = worker.process_next_task()

    assert processed is True
    job = jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    worker_trace = job["result"]["scene_narratives"][0]["trace"]["worker"]
    assert worker_trace["worker_id"] == "worker-test"
    assert worker_trace["attempt"] == 1
    task = queue.get_task(task_id=enqueued.task_id)
    assert task is not None
    assert task.status == QUEUE_STATUS_SUCCEEDED


def test_worker_retries_then_succeeds_after_transient_failure():
    queue = InMemorySceneTaskQueue()
    settings = _settings()
    job_id = jobs.create_job()
    enqueued = queue.enqueue_task(
        job_id=job_id,
        payload=_task_payload(job_id),
        idempotency_key=f"{job_id}:scene_understanding:v1",
        max_attempts=3,
    )
    worker = SceneAIWorker.from_settings(
        settings=settings,
        queue=queue,
        worker_id="worker-retry",
        media_store_factory=lambda: MagicMock(name="media_store"),
    )

    with patch(
        "app.scene_ai_worker.video_understanding.run_scene_understanding_pipeline",
        side_effect=[
            RuntimeError("temporary provider outage"),
            {"scene_narratives": [], "video_synopsis": None},
        ],
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
        idempotency_key=f"{job_id}:scene_understanding:v1",
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


def test_worker_fallback_policy_completes_with_empty_scene_outputs():
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
        idempotency_key=f"{job_id}:scene_understanding:v1",
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
    assert job["result"]["scene_narratives"] == []
    assert job["result"]["video_synopsis"] is None
    task = queue.get_task(task_id=enqueued.task_id)
    assert task is not None
    assert task.status == QUEUE_STATUS_DEAD_LETTER
