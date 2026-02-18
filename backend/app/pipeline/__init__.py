"""Pipeline package exports."""

from app.pipeline.aggregation import (
    aggregate_pipeline_frames,
    build_pipeline_metadata,
    build_result_payload,
)
from app.pipeline.contracts import (
    PipelineContext,
    PipelineStage,
    StageConfig,
    StageExecutionOutput,
    StageExecutionRecord,
    StageFailure,
)
from app.pipeline.executor import PipelineExecutionResult, PipelineExecutor
from app.pipeline.registry import PipelineStageRegistry, RegisteredStage, validate_stage_contract
from app.pipeline.stages import build_default_registry

__all__ = [
    "PipelineContext",
    "PipelineExecutionResult",
    "PipelineExecutor",
    "PipelineStage",
    "PipelineStageRegistry",
    "RegisteredStage",
    "StageConfig",
    "StageExecutionOutput",
    "StageExecutionRecord",
    "StageFailure",
    "aggregate_pipeline_frames",
    "build_default_registry",
    "build_pipeline_metadata",
    "build_result_payload",
    "validate_stage_contract",
]
