"""Deterministic stage-output aggregation for result payloads."""

from __future__ import annotations

from typing import Any

from app.pipeline.executor import PipelineExecutionResult

_CORE_ANALYSIS_KEYS = {
    "semantic_segmentation",
    "object_detection",
    "face_recognition",
    "enrichment",
}
_CORE_ANALYSIS_ARTIFACT_KEYS = {"json"}
_CORE_METADATA_KEYS = {"provenance", "model_provenance", "evidence_anchors"}


def _resolve_frame_branch_state(stage_status: str, error: str | None) -> dict[str, Any]:
    normalized = "success"
    if stage_status == "failed":
        normalized = "failed"
    elif stage_status == "skipped":
        normalized = "disabled"
    payload: dict[str, Any] = {"status": normalized}
    if error:
        payload["error"] = error
    return payload


def _resolve_chunk_branch_state(
    *,
    stage_status: str,
    skipped_reason: str | None,
    stage_error: str | None,
    chunk_payload: Any,
) -> dict[str, Any]:
    status = "success"
    error = stage_error

    if stage_status == "failed":
        status = "failed"
    elif stage_status == "skipped" and skipped_reason == "disabled":
        status = "disabled"
    elif isinstance(chunk_payload, dict):
        chunk_enabled = bool(chunk_payload.get("enabled"))
        chunk_error = chunk_payload.get("error")
        if isinstance(chunk_error, str) and chunk_error:
            status = "failed"
            error = chunk_error
        elif not chunk_enabled:
            status = "disabled"

    payload: dict[str, Any] = {"status": status}
    if error:
        payload["error"] = error
    return payload


def _build_branch_metadata(execution: PipelineExecutionResult) -> dict[str, Any]:
    status_by_stage = {record.stage_id: record for record in execution.stage_records}
    metadata: dict[str, Any] = {}

    frame_record = status_by_stage.get("frame_analysis")
    if frame_record is not None:
        metadata["frame_analysis"] = _resolve_frame_branch_state(
            frame_record.status,
            frame_record.error,
        )

    chunk_record = status_by_stage.get("parallel_chunked_tracking")
    if chunk_record is not None:
        chunk_payload = (
            execution.stage_outputs.get("parallel_chunked_tracking", {})
            .get("result_extensions", {})
            .get("video_chunked_tracks")
        )
        metadata["chunk_tracking"] = _resolve_chunk_branch_state(
            stage_status=chunk_record.status,
            skipped_reason=chunk_record.skipped_reason,
            stage_error=chunk_record.error,
            chunk_payload=chunk_payload,
        )

    return metadata


def _default_frame(frame_id: int) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "timestamp": "",
        "raw_frame_index": None,
        "files": {},
        "analysis": {
            "semantic_segmentation": [],
            "object_detection": [],
            "face_recognition": [],
            "enrichment": {},
            "extensions": {},
        },
        "analysis_artifacts": {
            "json": "",
            "extensions": {},
        },
        "metadata": {
            "provenance": {},
            "model_provenance": [],
            "evidence_anchors": [],
            "extensions": {},
        },
    }


def _merge_analysis(
    *,
    target: dict[str, Any],
    stage_id: str,
    source: dict[str, Any],
) -> None:
    extensions = target.setdefault("extensions", {})
    stage_ext = extensions.setdefault(stage_id, {})

    for key, value in source.items():
        if key in _CORE_ANALYSIS_KEYS:
            if stage_id == "identity_consistency" and key in {
                "object_detection",
                "face_recognition",
            }:
                target[key] = value
                continue
            if key not in target or target.get(key) in (None, "", [], {}):
                target[key] = value
                continue
            if target[key] != value:
                stage_ext[key] = value
            continue
        stage_ext[key] = value


def _merge_analysis_artifacts(
    *,
    target: dict[str, Any],
    stage_id: str,
    source: dict[str, Any],
) -> None:
    extensions = target.setdefault("extensions", {})
    stage_ext = extensions.setdefault(stage_id, {})

    for key, value in source.items():
        if key in _CORE_ANALYSIS_ARTIFACT_KEYS:
            if not target.get(key):
                target[key] = value
                continue
            if target[key] != value:
                stage_ext[key] = value
            continue
        stage_ext[key] = value


def _merge_metadata(
    *,
    target: dict[str, Any],
    stage_id: str,
    source: dict[str, Any],
) -> None:
    extensions = target.setdefault("extensions", {})
    stage_ext = extensions.setdefault(stage_id, {})

    for key, value in source.items():
        if key in _CORE_METADATA_KEYS:
            if key not in target or target.get(key) in (None, "", [], {}):
                target[key] = value
                continue
            if target[key] != value:
                stage_ext[key] = value
            continue
        stage_ext[key] = value


def aggregate_pipeline_frames(
    *,
    stage_order: list[str],
    frame_updates_by_stage: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate frame updates into deterministic frame payload sections."""
    frames_by_id: dict[int, dict[str, Any]] = {}
    for stage_id in stage_order:
        updates = frame_updates_by_stage.get(stage_id, {})
        for frame_id in sorted(updates):
            update = updates[frame_id]
            frame = frames_by_id.setdefault(frame_id, _default_frame(frame_id))

            timestamp = update.get("timestamp")
            if isinstance(timestamp, str) and timestamp:
                frame["timestamp"] = timestamp
            raw_frame_index = update.get("raw_frame_index")
            if isinstance(raw_frame_index, int) and not isinstance(
                raw_frame_index, bool
            ):
                frame["raw_frame_index"] = raw_frame_index

            files = update.get("files")
            if isinstance(files, dict):
                frame["files"].update(files)

            analysis = update.get("analysis")
            if isinstance(analysis, dict):
                _merge_analysis(target=frame["analysis"], stage_id=stage_id, source=analysis)

            analysis_artifacts = update.get("analysis_artifacts")
            if isinstance(analysis_artifacts, dict):
                _merge_analysis_artifacts(
                    target=frame["analysis_artifacts"],
                    stage_id=stage_id,
                    source=analysis_artifacts,
                )

            metadata = update.get("metadata")
            if isinstance(metadata, dict):
                _merge_metadata(target=frame["metadata"], stage_id=stage_id, source=metadata)

    return [frames_by_id[frame_id] for frame_id in sorted(frames_by_id)]


def build_pipeline_metadata(execution: PipelineExecutionResult) -> dict[str, Any]:
    """Build top-level pipeline metadata for API responses."""
    stages = [record.stage_id for record in execution.stage_records]
    return {
        "stages": stages,
        "status": [
            {
                "stage_id": record.stage_id,
                "status": record.status,
                "error": record.error,
                "skipped_reason": record.skipped_reason,
            }
            for record in execution.stage_records
        ],
        "failed_stage": execution.failed_stage_id,
    }


def build_result_payload(
    *,
    job_id: str,
    execution: PipelineExecutionResult,
) -> dict[str, Any]:
    """Build canonical result payload from modular pipeline execution."""
    stage_order = [record.stage_id for record in execution.stage_records]
    payload: dict[str, Any] = {
        "job_id": job_id,
        "pipeline": build_pipeline_metadata(execution),
        "branch_metadata": _build_branch_metadata(execution),
        "frames": aggregate_pipeline_frames(
            stage_order=stage_order,
            frame_updates_by_stage=execution.frame_updates_by_stage,
        ),
    }
    for stage_id in stage_order:
        stage_data = execution.stage_outputs.get(stage_id, {})
        if not isinstance(stage_data, dict):
            continue
        result_extensions = stage_data.get("result_extensions")
        if not isinstance(result_extensions, dict):
            continue
        for key, value in result_extensions.items():
            payload[str(key)] = value
    return payload
