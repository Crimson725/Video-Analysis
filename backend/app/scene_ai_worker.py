"""Out-of-process scene AI worker for queued scene-understanding tasks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from app import corpus, jobs, video_understanding
from app.scene_ai_worker_contracts import (
    SceneWorkerTaskInput,
    attach_worker_provenance,
    build_worker_provenance,
)
from app.scene_task_queue import (
    QUEUE_STATUS_DEAD_LETTER,
    QUEUE_STATUS_FAILED,
    SceneTask,
    SceneTaskQueue,
)
from app.storage import MediaStore, MediaStoreConfigError, MediaStoreError, R2MediaStore

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class NonRetryableSceneWorkerError(RuntimeError):
    """Error category for malformed/non-recoverable task failures."""


def _default_scene_outputs() -> dict[str, Any]:
    return {"scene_narratives": [], "video_synopsis": None}


def _normalize_scene_outputs(raw_scene_outputs: Any) -> dict[str, Any]:
    defaults = _default_scene_outputs()
    if not isinstance(raw_scene_outputs, dict):
        return defaults
    raw_scene_narratives = raw_scene_outputs.get("scene_narratives", [])
    if not isinstance(raw_scene_narratives, list):
        scene_narratives: list[dict[str, Any]] = []
    else:
        scene_narratives = [item for item in raw_scene_narratives if isinstance(item, dict)]
    video_synopsis = raw_scene_outputs.get("video_synopsis")
    if not isinstance(video_synopsis, dict):
        video_synopsis = None
    return {"scene_narratives": scene_narratives, "video_synopsis": video_synopsis}


def _build_media_store(settings: "Settings") -> MediaStore:
    return R2MediaStore(
        account_id=settings.r2_account_id,
        bucket=settings.r2_bucket,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        default_url_ttl_seconds=settings.r2_url_ttl_seconds,
    )


def _retry_backoff_seconds(settings: "Settings", attempt: int) -> int:
    base = max(1, int(settings.scene_ai_retry_backoff_seconds))
    multiplier = max(1, int(settings.scene_ai_retry_backoff_multiplier))
    max_backoff = max(1, int(settings.scene_ai_retry_backoff_max_seconds))
    backoff = base * (multiplier ** max(0, attempt - 1))
    return min(max_backoff, backoff)


@dataclass(slots=True)
class SceneAIWorker:
    """Worker process that claims and executes scene-understanding tasks."""

    settings: "Settings"
    queue: SceneTaskQueue
    worker_id: str
    media_store_factory: Callable[[], MediaStore]

    @classmethod
    def from_settings(
        cls,
        *,
        settings: "Settings",
        queue: SceneTaskQueue,
        worker_id: str | None = None,
        media_store_factory: Callable[[], MediaStore] | None = None,
    ) -> "SceneAIWorker":
        return cls(
            settings=settings,
            queue=queue,
            worker_id=worker_id or f"scene-worker-{uuid4()}",
            media_store_factory=media_store_factory or (lambda: _build_media_store(settings)),
        )

    def process_next_task(self) -> bool:
        """Claim one task and execute it. Returns True when a task was processed."""
        task = self.queue.claim_task(
            lease_owner=self.worker_id,
            lease_seconds=self.settings.scene_ai_lease_timeout_seconds,
        )
        if task is None:
            return False
        age_seconds = max(0.0, (datetime.now(UTC) - task.created_at).total_seconds())
        logger.info(
            "scene_ai_worker.claimed task_id=%s job_id=%s attempts=%s max_attempts=%s lease_owner=%s queue_age_seconds=%.3f",
            task.task_id,
            task.job_id,
            task.attempts,
            task.max_attempts,
            task.lease_owner,
            age_seconds,
        )
        self._execute_task(task)
        return True

    def run_forever(self) -> None:
        """Run polling loop until interrupted."""
        poll_seconds = max(1, int(self.settings.scene_ai_worker_poll_interval_seconds))
        logger.info("scene_ai_worker.start worker_id=%s poll_seconds=%s", self.worker_id, poll_seconds)
        while True:
            processed = self.process_next_task()
            if not processed:
                time.sleep(poll_seconds)

    def _execute_task(self, task: SceneTask) -> None:
        try:
            task_input = SceneWorkerTaskInput.from_payload(task.payload)
            if not task_input.job_id:
                raise NonRetryableSceneWorkerError("Task payload missing job_id")
            if not task_input.source_key:
                raise NonRetryableSceneWorkerError("Task payload missing source_key")

            media_store = self.media_store_factory()
            jobs.set_job_stage(task_input.job_id, "scene_ai_processing")
            scene_outputs = _default_scene_outputs()
            if self.settings.enable_scene_understanding_pipeline:
                scene_outputs = _normalize_scene_outputs(
                    video_understanding.run_scene_understanding_pipeline(
                        job_id=task_input.job_id,
                        scenes=task_input.scenes,
                        frame_results=task_input.frame_results,
                        settings=self.settings,
                        media_store=media_store,
                    )
                )
            provenance = build_worker_provenance(
                worker_id=self.worker_id,
                attempt=task.attempts,
                scene_model_id=self.settings.scene_model_id,
                synopsis_model_id=self.settings.synopsis_model_id,
                prompt_version=self.settings.scene_ai_prompt_version,
                runtime_version=self.settings.scene_ai_runtime_version,
            )
            scene_outputs = attach_worker_provenance(
                scene_outputs=scene_outputs,
                provenance=provenance,
            )
            payload = self._build_final_payload(
                task_input=task_input,
                scene_outputs=scene_outputs,
                media_store=media_store,
            )
            jobs.complete_job(task_input.job_id, payload)
            completed = self.queue.mark_succeeded(
                task_id=task.task_id,
                lease_owner=self.worker_id,
                result_metadata=provenance,
            )
            if completed is None:
                logger.info(
                    "scene_ai_worker.duplicate_completion task_id=%s job_id=%s",
                    task.task_id,
                    task_input.job_id,
                )
                return
            logger.info(
                "scene_ai_worker.succeeded task_id=%s job_id=%s attempts=%s",
                task.task_id,
                task_input.job_id,
                task.attempts,
            )
        except NonRetryableSceneWorkerError as exc:
            self._handle_non_retryable_error(task=task, error=str(exc))
        except (MediaStoreConfigError, MediaStoreError) as exc:
            self._handle_non_retryable_error(task=task, error=str(exc))
        except Exception as exc:  # pragma: no cover - exercised by failure tests
            self._handle_retryable_error(task=task, error=exc)

    def _build_final_payload(
        self,
        *,
        task_input: SceneWorkerTaskInput,
        scene_outputs: dict[str, Any],
        media_store: MediaStore,
    ) -> dict[str, Any]:
        corpus_output: dict[str, Any] | None = None
        if self.settings.enable_corpus_pipeline:
            jobs.set_job_stage(task_input.job_id, "corpus_processing")
            corpus_output = corpus.build(
                job_id=task_input.job_id,
                scenes=task_input.scenes,
                frame_results=task_input.frame_results,
                scene_outputs=scene_outputs,
                settings=self.settings,
                media_store=media_store,
            )
        return {
            "job_id": task_input.job_id,
            "frames": task_input.frame_results,
            **scene_outputs,
            "corpus": corpus_output,
            "video_face_identities": task_input.video_face_identities,
        }

    def _handle_non_retryable_error(self, *, task: SceneTask, error: str) -> None:
        transition = self.queue.mark_terminal_failure(
            task_id=task.task_id,
            lease_owner=self.worker_id,
            status=QUEUE_STATUS_DEAD_LETTER,
            last_error=error,
            error_metadata={
                "retryable": False,
                "attempt": task.attempts,
                "worker_id": self.worker_id,
            },
        )
        if transition is None:
            return
        self._handle_terminal_job_failure(task=task, error=error)

    def _handle_retryable_error(self, *, task: SceneTask, error: Exception) -> None:
        backoff_seconds = _retry_backoff_seconds(self.settings, task.attempts)
        next_attempt_at = datetime.now(UTC) + timedelta(seconds=backoff_seconds)
        transition = self.queue.mark_retry(
            task_id=task.task_id,
            lease_owner=self.worker_id,
            last_error=str(error),
            error_metadata={
                "retryable": True,
                "attempt": task.attempts,
                "backoff_seconds": backoff_seconds,
                "worker_id": self.worker_id,
            },
            next_attempt_at=next_attempt_at,
        )
        if transition is None:
            return
        if transition.status == QUEUE_STATUS_DEAD_LETTER:
            self._handle_terminal_job_failure(task=transition, error=str(error))
            logger.error(
                "scene_ai_worker.dead_letter task_id=%s job_id=%s attempts=%s error=%s",
                transition.task_id,
                transition.job_id,
                transition.attempts,
                error,
            )
            return
        jobs.set_job_stage(task.job_id, "waiting_scene_ai")
        logger.warning(
            "scene_ai_worker.retry_scheduled task_id=%s job_id=%s attempts=%s next_attempt_at=%s error=%s",
            task.task_id,
            task.job_id,
            task.attempts,
            next_attempt_at.isoformat(),
            error,
        )

    def _handle_terminal_job_failure(self, *, task: SceneTask, error: str) -> None:
        if self.settings.scene_ai_failure_policy == "fallback_empty":
            try:
                media_store = self.media_store_factory()
                task_input = SceneWorkerTaskInput.from_payload(task.payload)
                payload = self._build_final_payload(
                    task_input=task_input,
                    scene_outputs=_default_scene_outputs(),
                    media_store=media_store,
                )
                jobs.complete_job(task.job_id, payload)
                return
            except Exception as exc:  # pragma: no cover - defensive fallback path
                logger.exception(
                    "scene_ai_worker.fallback_failed task_id=%s job_id=%s error=%s",
                    task.task_id,
                    task.job_id,
                    exc,
                )
        self.queue.mark_terminal_failure(
            task_id=task.task_id,
            lease_owner=None,
            status=QUEUE_STATUS_FAILED,
            last_error=error,
            error_metadata={
                "retryable": False,
                "attempt": task.attempts,
                "worker_id": self.worker_id,
            },
        )
        jobs.fail_job(task.job_id, error)
