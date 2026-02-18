"""Unit tests for pipeline contracts, registry ordering, and executor isolation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.pipeline import (
    PipelineContext,
    PipelineExecutor,
    PipelineStageRegistry,
    StageExecutionOutput,
    validate_stage_contract,
)


@dataclass
class _DummySettings:
    cleanup_local_video_after_upload_default: bool = True


@dataclass
class _DummyModels:
    marker: str = "ok"


@dataclass
class _DummyMediaStore:
    marker: str = "ok"


def _context() -> PipelineContext:
    return PipelineContext(
        job_id="job-1",
        video_path="/tmp/video.mp4",
        source_extension="mp4",
        upload_content_type="video/mp4",
        local_dir="/tmp",
        settings=_DummySettings(),
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
