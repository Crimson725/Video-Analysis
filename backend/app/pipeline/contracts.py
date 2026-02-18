"""Pipeline stage contracts and execution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


StageId = str


@dataclass(slots=True)
class StageExecutionOutput:
    """Structured stage output with stage-scoped data and frame updates."""

    data: dict[str, Any] = field(default_factory=dict)
    frame_updates: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineContext:
    """Shared mutable context passed to all pipeline stages."""

    job_id: str
    video_path: str
    source_extension: str
    upload_content_type: str | None
    local_dir: str
    settings: Any
    models: Any
    media_store: Any
    source_key: str | None = None
    scenes: list[tuple[float, float]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    frame_results: list[dict[str, Any]] = field(default_factory=list)
    stage_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)


@runtime_checkable
class PipelineStage(Protocol):
    """Stage protocol implemented by all modular pipeline stages."""

    stage_id: StageId
    dependencies: tuple[StageId, ...]

    def run(self, context: PipelineContext) -> StageExecutionOutput:
        """Execute the stage and return stage-scoped output."""


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Registration configuration for one stage."""

    enabled: bool = True


@dataclass(frozen=True, slots=True)
class StageExecutionRecord:
    """Execution status for one stage invocation attempt."""

    stage_id: StageId
    status: str
    error: str | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StageFailure:
    """Structured failure details emitted by pipeline executor."""

    stage_id: StageId
    error: str
    exception_type: str
