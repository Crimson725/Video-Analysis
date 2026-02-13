"""Postgres-backed scene AI task queue with in-memory test implementation."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

QUEUE_STATUS_QUEUED = "queued"
QUEUE_STATUS_RETRY = "retry"
QUEUE_STATUS_PROCESSING = "processing"
QUEUE_STATUS_SUCCEEDED = "succeeded"
QUEUE_STATUS_FAILED = "failed"
QUEUE_STATUS_DEAD_LETTER = "dead_letter"

_ACTIVE_STATUSES = {
    QUEUE_STATUS_QUEUED,
    QUEUE_STATUS_RETRY,
    QUEUE_STATUS_PROCESSING,
}
_CLAIMABLE_STATUSES = {QUEUE_STATUS_QUEUED, QUEUE_STATUS_RETRY}

QUEUE_TASK_TYPE_SCENE_UNDERSTANDING = "scene_understanding"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scene_ai_tasks (
    id BIGSERIAL PRIMARY KEY,
    task_id UUID NOT NULL UNIQUE,
    job_id TEXT NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'scene_understanding',
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    lease_owner TEXT NULL,
    lease_expires_at TIMESTAMPTZ NULL,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    idempotency_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    result_metadata JSONB NULL,
    last_error TEXT NULL,
    error_metadata JSONB NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NULL
)
"""

CREATE_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_scene_ai_tasks_status_next_attempt
ON scene_ai_tasks (status, next_attempt_at)
"""

CREATE_LEASE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_scene_ai_tasks_lease_expires
ON scene_ai_tasks (lease_expires_at)
"""

CREATE_ACTIVE_IDEMPOTENCY_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_scene_ai_tasks_active_idempotency
ON scene_ai_tasks (idempotency_key)
WHERE status IN ('queued', 'retry', 'processing')
"""

ENQUEUE_TASK_SQL = """
INSERT INTO scene_ai_tasks (
    task_id,
    job_id,
    task_type,
    status,
    attempts,
    max_attempts,
    lease_owner,
    lease_expires_at,
    next_attempt_at,
    idempotency_key,
    payload,
    result_metadata,
    last_error,
    error_metadata,
    created_at,
    updated_at,
    completed_at
)
VALUES (
    %(task_id)s::uuid,
    %(job_id)s,
    %(task_type)s,
    %(status)s,
    0,
    %(max_attempts)s,
    NULL,
    NULL,
    %(next_attempt_at)s,
    %(idempotency_key)s,
    %(payload)s::jsonb,
    NULL,
    NULL,
    NULL,
    NOW(),
    NOW(),
    NULL
)
ON CONFLICT (idempotency_key) WHERE status IN ('queued', 'retry', 'processing')
DO NOTHING
RETURNING *
"""

SELECT_ACTIVE_BY_IDEMPOTENCY_SQL = """
SELECT *
FROM scene_ai_tasks
WHERE idempotency_key = %(idempotency_key)s
  AND status IN ('queued', 'retry', 'processing')
ORDER BY created_at ASC
LIMIT 1
"""

CLAIM_TASK_SQL = """
WITH candidate AS (
    SELECT id
    FROM scene_ai_tasks
    WHERE status IN ('queued', 'retry')
      AND next_attempt_at <= NOW()
      AND (lease_expires_at IS NULL OR lease_expires_at <= NOW())
    ORDER BY next_attempt_at ASC, id ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE scene_ai_tasks AS tasks
SET status = 'processing',
    attempts = tasks.attempts + 1,
    lease_owner = %(lease_owner)s,
    lease_expires_at = NOW() + make_interval(secs => %(lease_seconds)s),
    updated_at = NOW(),
    last_error = NULL,
    error_metadata = NULL
FROM candidate
WHERE tasks.id = candidate.id
RETURNING tasks.*
"""

MARK_SUCCEEDED_SQL = """
UPDATE scene_ai_tasks
SET status = 'succeeded',
    lease_owner = NULL,
    lease_expires_at = NULL,
    result_metadata = %(result_metadata)s::jsonb,
    updated_at = NOW(),
    completed_at = NOW()
WHERE task_id = %(task_id)s::uuid
  AND status = 'processing'
  AND (%(lease_owner)s IS NULL OR lease_owner = %(lease_owner)s)
RETURNING *
"""

MARK_RETRY_OR_DEAD_LETTER_SQL = """
UPDATE scene_ai_tasks
SET status = CASE
        WHEN attempts >= max_attempts THEN 'dead_letter'
        ELSE 'retry'
    END,
    lease_owner = NULL,
    lease_expires_at = NULL,
    next_attempt_at = CASE
        WHEN attempts >= max_attempts THEN next_attempt_at
        ELSE %(next_attempt_at)s
    END,
    last_error = %(last_error)s,
    error_metadata = %(error_metadata)s::jsonb,
    result_metadata = NULL,
    updated_at = NOW(),
    completed_at = CASE
        WHEN attempts >= max_attempts THEN NOW()
        ELSE NULL
    END
WHERE task_id = %(task_id)s::uuid
  AND status = 'processing'
  AND (%(lease_owner)s IS NULL OR lease_owner = %(lease_owner)s)
RETURNING *
"""

MARK_TERMINAL_SQL = """
UPDATE scene_ai_tasks
SET status = %(status)s,
    lease_owner = NULL,
    lease_expires_at = NULL,
    next_attempt_at = NOW(),
    last_error = %(last_error)s,
    error_metadata = %(error_metadata)s::jsonb,
    result_metadata = NULL,
    updated_at = NOW(),
    completed_at = NOW()
WHERE task_id = %(task_id)s::uuid
  AND status = 'processing'
  AND (%(lease_owner)s IS NULL OR lease_owner = %(lease_owner)s)
RETURNING *
"""

GET_TASK_SQL = """
SELECT *
FROM scene_ai_tasks
WHERE task_id = %(task_id)s::uuid
LIMIT 1
"""


@dataclass(slots=True)
class SceneTask:
    """Queue record for one scene-understanding task."""

    task_id: str
    job_id: str
    task_type: str
    status: str
    attempts: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    next_attempt_at: datetime
    idempotency_key: str
    payload: dict[str, Any]
    result_metadata: dict[str, Any] | None
    last_error: str | None
    error_metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class SceneTaskQueue(Protocol):
    """Queue operations used by API orchestration and worker runtime."""

    def ensure_schema(self) -> None:
        """Create queue schema objects if missing."""

    def enqueue_task(
        self,
        *,
        job_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        max_attempts: int,
    ) -> SceneTask:
        """Insert or return active task with matching idempotency key."""

    def claim_task(self, *, lease_owner: str, lease_seconds: int) -> SceneTask | None:
        """Claim one task using exclusive lease semantics."""

    def mark_succeeded(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        result_metadata: dict[str, Any],
    ) -> SceneTask | None:
        """Mark processing task as succeeded with idempotent CAS semantics."""

    def mark_retry(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        last_error: str,
        error_metadata: dict[str, Any],
        next_attempt_at: datetime,
    ) -> SceneTask | None:
        """Schedule retry or dead-letter transition after a failed attempt."""

    def mark_terminal_failure(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        status: str,
        last_error: str,
        error_metadata: dict[str, Any],
    ) -> SceneTask | None:
        """Mark processing task as terminal failure (`failed` or `dead_letter`)."""

    def get_task(self, *, task_id: str) -> SceneTask | None:
        """Fetch one task by task identifier."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _normalize_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_metadata(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    return None


class InMemorySceneTaskQueue:
    """In-memory queue used in tests."""

    def __init__(self) -> None:
        self._tasks: dict[str, SceneTask] = {}
        self._lock = threading.Lock()

    def ensure_schema(self) -> None:
        return

    def enqueue_task(
        self,
        *,
        job_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        max_attempts: int,
    ) -> SceneTask:
        now = _utcnow()
        with self._lock:
            for task in self._tasks.values():
                if task.idempotency_key == idempotency_key and task.status in _ACTIVE_STATUSES:
                    return task
            task = SceneTask(
                task_id=str(uuid4()),
                job_id=job_id,
                task_type=QUEUE_TASK_TYPE_SCENE_UNDERSTANDING,
                status=QUEUE_STATUS_QUEUED,
                attempts=0,
                max_attempts=max_attempts,
                lease_owner=None,
                lease_expires_at=None,
                next_attempt_at=now,
                idempotency_key=idempotency_key,
                payload=dict(payload),
                result_metadata=None,
                last_error=None,
                error_metadata=None,
                created_at=now,
                updated_at=now,
                completed_at=None,
            )
            self._tasks[task.task_id] = task
            return task

    def claim_task(self, *, lease_owner: str, lease_seconds: int) -> SceneTask | None:
        now = _utcnow()
        with self._lock:
            candidates = [
                task
                for task in self._tasks.values()
                if task.status in _CLAIMABLE_STATUSES
                and task.next_attempt_at <= now
                and (task.lease_expires_at is None or task.lease_expires_at <= now)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda task: (task.next_attempt_at, task.created_at, task.task_id))
            task = candidates[0]
            task.status = QUEUE_STATUS_PROCESSING
            task.attempts += 1
            task.lease_owner = lease_owner
            task.lease_expires_at = now + timedelta(seconds=max(1, lease_seconds))
            task.updated_at = now
            task.last_error = None
            task.error_metadata = None
            return task

    def mark_succeeded(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        result_metadata: dict[str, Any],
    ) -> SceneTask | None:
        now = _utcnow()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != QUEUE_STATUS_PROCESSING:
                return None
            if lease_owner is not None and task.lease_owner != lease_owner:
                return None
            task.status = QUEUE_STATUS_SUCCEEDED
            task.lease_owner = None
            task.lease_expires_at = None
            task.result_metadata = dict(result_metadata)
            task.updated_at = now
            task.completed_at = now
            return task

    def mark_retry(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        last_error: str,
        error_metadata: dict[str, Any],
        next_attempt_at: datetime,
    ) -> SceneTask | None:
        now = _utcnow()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != QUEUE_STATUS_PROCESSING:
                return None
            if lease_owner is not None and task.lease_owner != lease_owner:
                return None
            if task.attempts >= task.max_attempts:
                task.status = QUEUE_STATUS_DEAD_LETTER
                task.completed_at = now
            else:
                task.status = QUEUE_STATUS_RETRY
                task.next_attempt_at = next_attempt_at.astimezone(UTC)
                task.completed_at = None
            task.lease_owner = None
            task.lease_expires_at = None
            task.last_error = last_error
            task.error_metadata = dict(error_metadata)
            task.result_metadata = None
            task.updated_at = now
            return task

    def mark_terminal_failure(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        status: str,
        last_error: str,
        error_metadata: dict[str, Any],
    ) -> SceneTask | None:
        if status not in {QUEUE_STATUS_FAILED, QUEUE_STATUS_DEAD_LETTER}:
            raise ValueError(f"Unsupported terminal status: {status}")
        now = _utcnow()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != QUEUE_STATUS_PROCESSING:
                return None
            if lease_owner is not None and task.lease_owner != lease_owner:
                return None
            task.status = status
            task.lease_owner = None
            task.lease_expires_at = None
            task.next_attempt_at = now
            task.last_error = last_error
            task.error_metadata = dict(error_metadata)
            task.result_metadata = None
            task.updated_at = now
            task.completed_at = now
            return task

    def get_task(self, *, task_id: str) -> SceneTask | None:
        with self._lock:
            return self._tasks.get(task_id)


class PostgresSceneTaskQueue:
    """Postgres-backed queue implementation."""

    def __init__(self, *, dsn: str, connect_factory: Any | None = None) -> None:
        self._dsn = dsn
        self._connect_factory = connect_factory

    def _connect(self) -> Any:
        if self._connect_factory is not None:
            return self._connect_factory()
        try:
            import psycopg
            from psycopg.rows import dict_row
        except Exception as exc:  # pragma: no cover - import availability varies by env
            raise RuntimeError(
                "psycopg is required for queue mode. Install psycopg[binary] to enable Postgres queue support."
            ) from exc
        return psycopg.connect(self._dsn, row_factory=dict_row)

    @staticmethod
    def _row_to_task(row: dict[str, Any]) -> SceneTask:
        return SceneTask(
            task_id=str(row["task_id"]),
            job_id=str(row["job_id"]),
            task_type=str(row["task_type"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            lease_owner=row.get("lease_owner"),
            lease_expires_at=row.get("lease_expires_at"),
            next_attempt_at=row["next_attempt_at"],
            idempotency_key=str(row["idempotency_key"]),
            payload=_normalize_payload(row.get("payload")),
            result_metadata=_normalize_metadata(row.get("result_metadata")),
            last_error=row.get("last_error"),
            error_metadata=_normalize_metadata(row.get("error_metadata")),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row.get("completed_at"),
        )

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(CREATE_STATUS_INDEX_SQL)
                cur.execute(CREATE_LEASE_INDEX_SQL)
                cur.execute(CREATE_ACTIVE_IDEMPOTENCY_INDEX_SQL)
            conn.commit()

    def enqueue_task(
        self,
        *,
        job_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        max_attempts: int,
    ) -> SceneTask:
        params = {
            "task_id": str(uuid4()),
            "job_id": job_id,
            "task_type": QUEUE_TASK_TYPE_SCENE_UNDERSTANDING,
            "status": QUEUE_STATUS_QUEUED,
            "max_attempts": max(1, max_attempts),
            "next_attempt_at": _utcnow(),
            "idempotency_key": idempotency_key,
            "payload": json.dumps(payload, separators=(",", ":")),
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ENQUEUE_TASK_SQL, params)
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        SELECT_ACTIVE_BY_IDEMPOTENCY_SQL,
                        {"idempotency_key": idempotency_key},
                    )
                    row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to enqueue or fetch active scene AI task")
            conn.commit()
        return self._row_to_task(row)

    def claim_task(self, *, lease_owner: str, lease_seconds: int) -> SceneTask | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    CLAIM_TASK_SQL,
                    {
                        "lease_owner": lease_owner,
                        "lease_seconds": max(1, int(lease_seconds)),
                    },
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return self._row_to_task(row)

    def mark_succeeded(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        result_metadata: dict[str, Any],
    ) -> SceneTask | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    MARK_SUCCEEDED_SQL,
                    {
                        "task_id": task_id,
                        "lease_owner": lease_owner,
                        "result_metadata": json.dumps(result_metadata, separators=(",", ":")),
                    },
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return self._row_to_task(row)

    def mark_retry(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        last_error: str,
        error_metadata: dict[str, Any],
        next_attempt_at: datetime,
    ) -> SceneTask | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    MARK_RETRY_OR_DEAD_LETTER_SQL,
                    {
                        "task_id": task_id,
                        "lease_owner": lease_owner,
                        "last_error": last_error,
                        "error_metadata": json.dumps(error_metadata, separators=(",", ":")),
                        "next_attempt_at": next_attempt_at.astimezone(UTC),
                    },
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return self._row_to_task(row)

    def mark_terminal_failure(
        self,
        *,
        task_id: str,
        lease_owner: str | None,
        status: str,
        last_error: str,
        error_metadata: dict[str, Any],
    ) -> SceneTask | None:
        if status not in {QUEUE_STATUS_FAILED, QUEUE_STATUS_DEAD_LETTER}:
            raise ValueError(f"Unsupported terminal status: {status}")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    MARK_TERMINAL_SQL,
                    {
                        "task_id": task_id,
                        "lease_owner": lease_owner,
                        "status": status,
                        "last_error": last_error,
                        "error_metadata": json.dumps(error_metadata, separators=(",", ":")),
                    },
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return self._row_to_task(row)

    def get_task(self, *, task_id: str) -> SceneTask | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_TASK_SQL, {"task_id": task_id})
                row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_task(row)


def build_postgres_scene_task_queue(*, dsn: str) -> PostgresSceneTaskQueue:
    """Construct the production Postgres queue implementation."""
    return PostgresSceneTaskQueue(dsn=dsn)
