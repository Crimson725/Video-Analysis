"""Default modular pipeline stage implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app import analysis, parallel_tracking, scene
from app.pipeline.artifacts import collect_required_artifact_keys, verify_required_artifacts
from app.pipeline.contracts import PipelineContext, PipelineStage, StageExecutionOutput

if TYPE_CHECKING:
    from app.pipeline.registry import PipelineStageRegistry

logger = logging.getLogger(__name__)


class SourceUploadStage:
    """Upload source video and validate source artifact availability."""

    stage_id = "source_upload"
    dependencies: tuple[str, ...] = ()

    def run(self, context: PipelineContext) -> StageExecutionOutput:
        source_key = context.media_store.upload_source_video(
            job_id=context.job_id,
            file_path=context.video_path,
            content_type=_resolve_video_content_type(
                context.upload_content_type, context.source_extension
            ),
            source_extension=context.source_extension,
        )
        source_upload_verified = context.media_store.verify_object(source_key)
        if not source_upload_verified:
            raise RuntimeError(f"Source upload verification failed for key '{source_key}'")
        context.source_key = source_key
        return StageExecutionOutput(data={"source_key": source_key})


class SceneDetectionStage:
    """Detect scenes in the uploaded source video."""

    stage_id = "scene_detection"
    dependencies: tuple[str, ...] = ("source_upload",)

    def run(self, context: PipelineContext) -> StageExecutionOutput:
        context.scenes = scene.detect_scenes(context.video_path)
        return StageExecutionOutput(data={"scene_count": len(context.scenes)})


class KeyframeExtractionStage:
    """Extract keyframes for scene-centered analysis."""

    stage_id = "keyframe_extraction"
    dependencies: tuple[str, ...] = ("scene_detection",)

    def run(self, context: PipelineContext) -> StageExecutionOutput:
        context.frames = scene.extract_keyframes(context.video_path, context.scenes)
        if not context.frames:
            raise RuntimeError("No scenes or frames extracted")
        return StageExecutionOutput(data={"frame_count": len(context.frames)})


class SaveOriginalFramesStage:
    """Persist original keyframes before analysis stages run."""

    stage_id = "save_original_frames"
    dependencies: tuple[str, ...] = ("keyframe_extraction",)

    def run(self, context: PipelineContext) -> StageExecutionOutput:
        scene.save_original_frames(
            context.frames,
            context.job_id,
            context.local_dir,
            media_store=context.media_store,
        )
        return StageExecutionOutput()


class FrameAnalysisStage:
    """Run CV analysis on each extracted keyframe."""

    stage_id = "frame_analysis"
    dependencies: tuple[str, ...] = ("save_original_frames",)

    def run(self, context: PipelineContext) -> StageExecutionOutput:
        frame_results: list[dict[str, Any]] = []
        face_tracker = analysis.FaceIdentityTracker()
        object_tracker = analysis.ObjectTrackTracker()
        for frame_data in context.frames:
            result = analysis.analyze_frame(
                frame_data,
                context.models,
                context.job_id,
                context.local_dir,
                media_store=context.media_store,
                face_tracker=face_tracker,
                object_tracker=object_tracker,
            )
            frame_results.append(result)
        context.frame_results = frame_results

        frame_updates: dict[int, dict[str, Any]] = {}
        for frame in frame_results:
            frame_id = int(frame.get("frame_id", 0))
            frame_updates[frame_id] = {
                "timestamp": str(frame.get("timestamp", "")),
                "raw_frame_index": frame.get("raw_frame_index"),
                "files": dict(frame.get("files", {})),
                "analysis": dict(frame.get("analysis", {})),
                "analysis_artifacts": dict(frame.get("analysis_artifacts", {})),
                "metadata": dict(frame.get("metadata", {})),
            }

        return StageExecutionOutput(
            data={"frame_count": len(frame_results)},
            frame_updates=frame_updates,
        )


class ArtifactVerificationStage:
    """Verify persisted frame and analysis artifacts before completion."""

    stage_id = "artifact_verification"
    dependencies: tuple[str, ...] = ("source_upload", "frame_analysis")

    def run(self, context: PipelineContext) -> StageExecutionOutput:
        source_key = context.source_key or ""
        payload = {
            "job_id": context.job_id,
            "frames": context.frame_results,
        }
        required_keys = collect_required_artifact_keys(context.job_id, payload, source_key)
        verify_required_artifacts(context.media_store, context.job_id, required_keys)
        return StageExecutionOutput(data={"required_artifact_count": len(required_keys)})


class ParallelChunkedTrackingStage:
    """Run the chunked tracking branch independently from keyframe extraction."""

    stage_id = "parallel_chunked_tracking"
    dependencies: tuple[str, ...] = ("source_upload", "scene_detection")

    def run(self, context: PipelineContext) -> StageExecutionOutput:
        enabled = bool(
            getattr(context.settings, "enable_parallel_chunked_tracking_pipeline", False)
        )
        if not enabled:
            payload = {
                "enabled": False,
                "method": "chunked_botsort_stitch_v1",
                "output_mode": getattr(
                    context.settings, "parallel_tracking_output_mode", "summary_v2"
                ),
                "backend_strategy": getattr(
                    context.settings, "parallel_tracking_backend_strategy", "default"
                ),
                "stitch_strategy": getattr(
                    context.settings, "parallel_tracking_stitch_strategy", "default"
                ),
                "zone_strategy": getattr(
                    context.settings, "parallel_tracking_zone_strategy", "grid3x3"
                ),
                "tracks": [],
                "scenes": [],
                "zone_definition": None,
                "entities": [],
                "stats": {
                    "chunk_count": 0,
                    "row_count": 0,
                    "track_count": 0,
                    "scene_count": 0,
                    "span_count": 0,
                    "key_box_count": 0,
                },
                "artifacts": {},
            }
            return StageExecutionOutput(
                data={"result_extensions": {"video_chunked_tracks": payload}}
            )
        try:
            payload = parallel_tracking.run_parallel_chunked_tracking(
                video_path=context.video_path,
                settings=context.settings,
                scenes=context.scenes,
                output_dir=f"{context.local_dir}/{context.job_id}/tracking",
            )
        except Exception as exc:  # pragma: no cover - defensive safety path
            logger.exception(
                "parallel_chunked_tracking stage failed job_id=%s", context.job_id
            )
            payload = {
                "enabled": False,
                "method": "chunked_botsort_stitch_v1",
                "output_mode": getattr(
                    context.settings, "parallel_tracking_output_mode", "summary_v2"
                ),
                "backend_strategy": getattr(
                    context.settings, "parallel_tracking_backend_strategy", "default"
                ),
                "stitch_strategy": getattr(
                    context.settings, "parallel_tracking_stitch_strategy", "default"
                ),
                "zone_strategy": getattr(
                    context.settings, "parallel_tracking_zone_strategy", "grid3x3"
                ),
                "tracks": [],
                "scenes": [],
                "zone_definition": None,
                "entities": [],
                "stats": {
                    "chunk_count": 0,
                    "row_count": 0,
                    "track_count": 0,
                    "scene_count": 0,
                    "span_count": 0,
                    "key_box_count": 0,
                },
                "artifacts": {},
                "error": str(exc),
            }
        return StageExecutionOutput(
            data={"result_extensions": {"video_chunked_tracks": payload}}
        )


def _resolve_video_content_type(
    upload_content_type: str | None,
    source_extension: str,
) -> str:
    if upload_content_type and "/" in upload_content_type:
        return upload_content_type
    if source_extension == "mp4":
        return "video/mp4"
    return f"video/{source_extension}"


def build_default_registry(
    *,
    enablement: dict[str, bool] | None = None,
) -> PipelineStageRegistry:
    """Build default registry for modular video analysis execution."""
    from app.pipeline.registry import PipelineStageRegistry

    enablement = enablement or {}
    registry = PipelineStageRegistry()
    stages: tuple[PipelineStage, ...] = (
        SourceUploadStage(),
        SceneDetectionStage(),
        KeyframeExtractionStage(),
        SaveOriginalFramesStage(),
        FrameAnalysisStage(),
        ParallelChunkedTrackingStage(),
        ArtifactVerificationStage(),
    )
    for stage in stages:
        registry.register(stage, enabled=enablement.get(stage.stage_id, True))
    return registry
