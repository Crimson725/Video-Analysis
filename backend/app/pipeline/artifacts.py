"""Helpers for required artifact key collection and verification."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from app.storage import MediaStore, MediaStoreError

logger = logging.getLogger(__name__)


def is_jobs_object_key(object_key: Any) -> bool:
    """Check whether a value is a persisted artifact object key."""
    return isinstance(object_key, str) and object_key.startswith("jobs/")


def _add_required_key_or_log(
    *,
    required: set[str],
    object_key: Any,
    warning_template: str,
    warning_context: tuple[Any, ...],
) -> None:
    if is_jobs_object_key(object_key):
        required.add(object_key)
        return
    logger.warning(warning_template, *warning_context, object_key)


def _collect_required_keys_from_values(
    *,
    required: set[str],
    values: Iterable[Any],
    warning_template: str,
    warning_context: tuple[Any, ...],
) -> None:
    for object_key in values:
        _add_required_key_or_log(
            required=required,
            object_key=object_key,
            warning_template=warning_template,
            warning_context=warning_context,
        )


def collect_required_artifact_keys(
    job_id: str,
    result_payload: dict[str, Any],
    source_key: str,
) -> set[str]:
    """Collect required object keys that must verify before job completion."""
    required: set[str] = {source_key}
    for frame in result_payload.get("frames", []):
        frame_id = frame.get("frame_id")
        _collect_required_keys_from_values(
            required=required,
            values=frame.get("files", {}).values(),
            warning_template="upload.verify.invalid_frame_file_key job_id=%s frame_id=%s value=%s",
            warning_context=(job_id, frame_id),
        )
        _collect_required_keys_from_values(
            required=required,
            values=frame.get("analysis_artifacts", {}).values(),
            warning_template="upload.verify.invalid_analysis_key job_id=%s frame_id=%s value=%s",
            warning_context=(job_id, frame_id),
        )
    return required


def verify_required_artifacts(
    media_store: MediaStore,
    job_id: str,
    required_keys: set[str],
) -> None:
    """Verify required objects exist in storage before marking a job complete."""
    missing = sorted(key for key in required_keys if not media_store.verify_object(key))
    if missing:
        preview = ", ".join(missing[:5])
        logger.error(
            "upload.verify.failed job_id=%s missing_count=%s sample=%s",
            job_id,
            len(missing),
            preview,
        )
        raise MediaStoreError(
            f"Upload verification failed for {len(missing)} artifact(s); sample: {preview}"
        )
    logger.info(
        "upload.verify.success job_id=%s artifact_count=%s", job_id, len(required_keys)
    )
