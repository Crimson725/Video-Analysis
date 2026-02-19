"""Unit tests for pipeline contracts, registry ordering, and executor isolation."""

from __future__ import annotations

from dataclasses import dataclass
import time

import pytest

from app.pipeline import (
    PipelineContext,
    PipelineExecutor,
    PipelineStageRegistry,
    StageExecutionOutput,
    build_result_payload,
    validate_stage_contract,
)


@dataclass
class _DummySettings:
    cleanup_local_video_after_upload_default: bool = True
    enable_pipeline_branch_concurrency: bool = False
    pipeline_frame_branch_worker_budget: int = 1
    pipeline_chunk_branch_worker_budget: int = 1


@dataclass
class _DummyModels:
    marker: str = "ok"


@dataclass
class _DummyMediaStore:
    marker: str = "ok"


def _context(settings: _DummySettings | None = None) -> PipelineContext:
    return PipelineContext(
        job_id="job-1",
        video_path="/tmp/video.mp4",
        source_extension="mp4",
        upload_content_type="video/mp4",
        local_dir="/tmp",
        settings=settings or _DummySettings(),
        models=_DummyModels(),
        media_store=_DummyMediaStore(),
    )


def test_validate_stage_contract_rejects_invalid_stage_id() -> None:
    class InvalidStage:
        stage_id = ""
        dependencies = ()

        def run(self, context):
            return StageExecutionOutput()

    with pytest.raises(ValueError, match="stage_id"):
        validate_stage_contract(InvalidStage())


def test_registry_orders_stages_deterministically_with_dependencies() -> None:
    class StageA:
        stage_id = "a"
        dependencies = ()

        def run(self, context):
            return StageExecutionOutput(data={"a": True})

    class StageB:
        stage_id = "b"
        dependencies = ("a",)

        def run(self, context):
            return StageExecutionOutput(data={"b": True})

    class StageC:
        stage_id = "c"
        dependencies = ("a",)

        def run(self, context):
            return StageExecutionOutput(data={"c": True})

    registry = PipelineStageRegistry()
    registry.register(StageB(), enabled=True)
    registry.register(StageA(), enabled=True)
    registry.register(StageC(), enabled=False)

    ordered = registry.ordered()

    assert [item.stage.stage_id for item in ordered] == ["a", "b", "c"]
    assert ordered[2].enabled is False


def test_executor_skips_dependent_stages_after_failure() -> None:
    class StageA:
        stage_id = "a"
        dependencies = ()

        def run(self, context):
            return StageExecutionOutput(data={"a": 1})

    class StageB:
        stage_id = "b"
        dependencies = ("a",)

        def run(self, context):
            raise RuntimeError("boom")

    class StageC:
        stage_id = "c"
        dependencies = ("b",)

        def run(self, context):
            return StageExecutionOutput(data={"c": 1})

    class StageD:
        stage_id = "d"
        dependencies = ()

        def run(self, context):
            return StageExecutionOutput(data={"d": 1})

    registry = PipelineStageRegistry()
    registry.register(StageA())
    registry.register(StageB())
    registry.register(StageC())
    registry.register(StageD())

    execution = PipelineExecutor().execute(context=_context(), registry=registry)

    assert execution.failed_stage_id == "b"
    status_by_stage = {record.stage_id: record.status for record in execution.stage_records}
    skip_reason_by_stage = {
        record.stage_id: record.skipped_reason for record in execution.stage_records
    }
    assert status_by_stage["a"] == "success"
    assert status_by_stage["b"] == "failed"
    assert status_by_stage["c"] == "skipped"
    assert skip_reason_by_stage["c"] == "dependency_failed"
    assert status_by_stage["d"] == "success"


def test_executor_runs_independent_branches_concurrently_when_enabled() -> None:
    timings: dict[str, tuple[float, float]] = {}

    class Source:
        stage_id = "source_upload"
        dependencies = ()

        def run(self, context):
            del context
            return StageExecutionOutput()

    class Scene:
        stage_id = "scene_detection"
        dependencies = ("source_upload",)

        def run(self, context):
            del context
            return StageExecutionOutput()

    class Keyframe:
        stage_id = "keyframe_extraction"
        dependencies = ("scene_detection",)

        def run(self, context):
            del context
            time.sleep(0.03)
            return StageExecutionOutput()

    class Frame:
        stage_id = "frame_analysis"
        dependencies = ("keyframe_extraction",)

        def run(self, context):
            del context
            started = time.perf_counter()
            time.sleep(0.12)
            ended = time.perf_counter()
            timings["frame"] = (started, ended)
            return StageExecutionOutput()

    class Chunk:
        stage_id = "parallel_chunked_tracking"
        dependencies = ("scene_detection",)

        def run(self, context):
            del context
            started = time.perf_counter()
            time.sleep(0.12)
            ended = time.perf_counter()
            timings["chunk"] = (started, ended)
            return StageExecutionOutput(
                data={
                    "result_extensions": {
                        "video_chunked_tracks": {"enabled": True, "entities": []}
                    }
                }
            )

    registry = PipelineStageRegistry()
    registry.register(Source())
    registry.register(Scene())
    registry.register(Keyframe())
    registry.register(Frame())
    registry.register(Chunk())

    execution = PipelineExecutor().execute(
        context=_context(
            _DummySettings(
                enable_pipeline_branch_concurrency=True,
                pipeline_frame_branch_worker_budget=1,
                pipeline_chunk_branch_worker_budget=1,
            )
        ),
        registry=registry,
    )

    assert execution.failed_stage_id is None
    frame_start, frame_end = timings["frame"]
    chunk_start, chunk_end = timings["chunk"]
    overlap = min(frame_end, chunk_end) - max(frame_start, chunk_start)
    assert overlap > 0.04


def test_result_payload_preserves_branch_isolation_and_branch_metadata() -> None:
    class Frame:
        stage_id = "frame_analysis"
        dependencies = ()

        def run(self, context):
            del context
            return StageExecutionOutput(
                frame_updates={
                    0: {
                        "analysis": {
                            "semantic_segmentation": [],
                            "object_detection": [{"label": "person"}],
                            "face_recognition": [],
                        }
                    }
                }
            )

    class Chunk:
        stage_id = "parallel_chunked_tracking"
        dependencies = ()

        def run(self, context):
            del context
            return StageExecutionOutput(
                data={
                    "result_extensions": {
                        "video_chunked_tracks": {
                            "enabled": True,
                            "method": "chunked_botsort_stitch_v1",
                            "output_mode": "summary_v2",
                            "zone_definition": {
                                "layout": "3x3",
                                "frame_width": 100,
                                "frame_height": 100,
                                "labels": ["top-left"],
                                "zones": {"top-left": {"x1": 0, "y1": 0, "x2": 33, "y2": 33}},
                            },
                            "entities": [],
                        }
                    }
                }
            )

    registry = PipelineStageRegistry()
    registry.register(Frame())
    registry.register(Chunk())

    execution = PipelineExecutor().execute(context=_context(), registry=registry)
    payload = build_result_payload(job_id="job-1", execution=execution)

    assert payload["frames"][0]["analysis"]["object_detection"][0]["label"] == "person"
    assert payload["video_chunked_tracks"]["enabled"] is True
    assert payload["branch_metadata"]["frame_analysis"]["status"] == "success"
    assert payload["branch_metadata"]["chunk_tracking"]["status"] == "success"


def test_branch_metadata_captures_disabled_and_failed_chunk_outcomes() -> None:
    class Frame:
        stage_id = "frame_analysis"
        dependencies = ()

        def run(self, context):
            del context
            return StageExecutionOutput()

    class Chunk:
        stage_id = "parallel_chunked_tracking"
        dependencies = ()

        def __init__(self, *, fail: bool) -> None:
            self._fail = fail

        def run(self, context):
            del context
            if self._fail:
                return StageExecutionOutput(
                    data={
                        "result_extensions": {
                            "video_chunked_tracks": {
                                "enabled": False,
                                "error": "chunk stage failed",
                            }
                        }
                    }
                )
            return StageExecutionOutput(
                data={"result_extensions": {"video_chunked_tracks": {"enabled": False}}}
            )

    failed_registry = PipelineStageRegistry()
    failed_registry.register(Frame())
    failed_registry.register(Chunk(fail=True))
    failed_execution = PipelineExecutor().execute(
        context=_context(),
        registry=failed_registry,
    )
    failed_payload = build_result_payload(job_id="job-1", execution=failed_execution)
    assert failed_payload["branch_metadata"]["frame_analysis"]["status"] == "success"
    assert failed_payload["branch_metadata"]["chunk_tracking"]["status"] == "failed"
    assert (
        failed_payload["branch_metadata"]["chunk_tracking"]["error"]
        == "chunk stage failed"
    )

    disabled_registry = PipelineStageRegistry()
    disabled_registry.register(Frame())
    disabled_registry.register(Chunk(fail=False), enabled=False)
    disabled_execution = PipelineExecutor().execute(
        context=_context(),
        registry=disabled_registry,
    )
    disabled_payload = build_result_payload(job_id="job-1", execution=disabled_execution)
    assert disabled_payload["branch_metadata"]["chunk_tracking"]["status"] == "disabled"
