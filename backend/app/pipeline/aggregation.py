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


def _default_frame(frame_id: int) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "timestamp": "",
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
    return {
        "job_id": job_id,
        "pipeline": build_pipeline_metadata(execution),
        "frames": aggregate_pipeline_frames(
            stage_order=stage_order,
            frame_updates_by_stage=execution.frame_updates_by_stage,
        ),
    }
