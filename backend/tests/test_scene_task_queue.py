"""Unit tests for scene task queue semantics."""

from datetime import UTC, datetime, timedelta

from app.scene_task_queue import (
    CLAIM_TASK_SQL,
    QUEUE_STATUS_DEAD_LETTER,
    QUEUE_STATUS_PROCESSING,
    QUEUE_STATUS_QUEUED,
    QUEUE_STATUS_RETRY,
    QUEUE_STATUS_SUCCEEDED,
    InMemorySceneTaskQueue,
)


def test_claim_query_uses_skip_locked():
    assert "FOR UPDATE SKIP LOCKED" in CLAIM_TASK_SQL


def test_inmemory_queue_state_transitions_retry_to_dead_letter():
    queue = InMemorySceneTaskQueue()
    task = queue.enqueue_task(
        job_id="job-1",
        payload={"job_id": "job-1"},
        idempotency_key="job-1:scene_understanding:v1",
        max_attempts=2,
    )
    assert task.status == QUEUE_STATUS_QUEUED

    claimed_one = queue.claim_task(lease_owner="worker-a", lease_seconds=30)
    assert claimed_one is not None
    assert claimed_one.status == QUEUE_STATUS_PROCESSING
    assert claimed_one.attempts == 1

    retried = queue.mark_retry(
        task_id=claimed_one.task_id,
        lease_owner="worker-a",
        last_error="transient timeout",
        error_metadata={"retryable": True},
        next_attempt_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    assert retried is not None
    assert retried.status == QUEUE_STATUS_RETRY
    assert retried.attempts == 1

    claimed_two = queue.claim_task(lease_owner="worker-a", lease_seconds=30)
    assert claimed_two is not None
    assert claimed_two.task_id == task.task_id
    assert claimed_two.attempts == 2

    dead_lettered = queue.mark_retry(
        task_id=claimed_two.task_id,
        lease_owner="worker-a",
        last_error="still failing",
        error_metadata={"retryable": True},
        next_attempt_at=datetime.now(UTC) + timedelta(seconds=30),
    )
    assert dead_lettered is not None
    assert dead_lettered.status == QUEUE_STATUS_DEAD_LETTER
    assert dead_lettered.completed_at is not None


def test_inmemory_enqueue_is_idempotent_for_active_task():
    queue = InMemorySceneTaskQueue()
    first = queue.enqueue_task(
        job_id="job-2",
        payload={"job_id": "job-2"},
        idempotency_key="job-2:scene_understanding:v1",
        max_attempts=3,
    )
    second = queue.enqueue_task(
        job_id="job-2",
        payload={"job_id": "job-2"},
        idempotency_key="job-2:scene_understanding:v1",
        max_attempts=3,
    )
    assert first.task_id == second.task_id


def test_mark_succeeded_is_compare_and_set():
    queue = InMemorySceneTaskQueue()
    task = queue.enqueue_task(
        job_id="job-3",
        payload={"job_id": "job-3"},
        idempotency_key="job-3:scene_understanding:v1",
        max_attempts=3,
    )
    claimed = queue.claim_task(lease_owner="worker-a", lease_seconds=30)
    assert claimed is not None
    assert claimed.task_id == task.task_id

    done = queue.mark_succeeded(
        task_id=task.task_id,
        lease_owner="worker-a",
        result_metadata={"attempt": 1},
    )
    assert done is not None
    assert done.status == QUEUE_STATUS_SUCCEEDED

    duplicate = queue.mark_succeeded(
        task_id=task.task_id,
        lease_owner="worker-a",
        result_metadata={"attempt": 1},
    )
    assert duplicate is None
