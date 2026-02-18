"""Deterministic modular pipeline executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.pipeline.contracts import (
    PipelineContext,
    StageExecutionRecord,
    StageFailure,
)
from app.pipeline.registry import PipelineStageRegistry, RegisteredStage


StageCallback = Callable[[str], None]


@dataclass(slots=True)
class PipelineExecutionResult:
    """Executor output with stage records, outputs, and failure metadata."""

    stage_outputs: dict[str, dict]
    frame_updates_by_stage: dict[str, dict[int, dict]]
    stage_records: list[StageExecutionRecord] = field(default_factory=list)
    failure: StageFailure | None = None

    @property
    def failed_stage_id(self) -> str | None:
        if self.failure is None:
            return None
        return self.failure.stage_id

    @property
    def is_success(self) -> bool:
        return self.failure is None


class PipelineExecutor:
    """Run registry stages using shared context and stage-scoped namespaces."""

    def execute(
        self,
        *,
        context: PipelineContext,
        registry: PipelineStageRegistry,
        on_stage_started: StageCallback | None = None,
    ) -> PipelineExecutionResult:
        stage_outputs: dict[str, dict] = {}
        frame_updates_by_stage: dict[str, dict[int, dict]] = {}
        stage_records: list[StageExecutionRecord] = []
        failed_or_blocked: set[str] = set()
        failure: StageFailure | None = None

        for registered in registry.ordered():
            stage_id = registered.stage.stage_id
            if not registered.enabled:
                stage_records.append(
                    StageExecutionRecord(
                        stage_id=stage_id,
                        status="skipped",
                        skipped_reason="disabled",
                    )
                )
                continue

            if any(dep in failed_or_blocked for dep in registered.stage.dependencies):
                failed_or_blocked.add(stage_id)
                stage_records.append(
                    StageExecutionRecord(
                        stage_id=stage_id,
                        status="skipped",
                        skipped_reason="dependency_failed",
                    )
                )
                continue

            if on_stage_started is not None:
                on_stage_started(stage_id)

            try:
                output = registered.stage.run(context)
            except Exception as exc:  # pragma: no cover - validated in unit tests
                if failure is None:
                    failure = StageFailure(
                        stage_id=stage_id,
                        error=str(exc),
                        exception_type=type(exc).__name__,
                    )
                failed_or_blocked.add(stage_id)
                stage_records.append(
                    StageExecutionRecord(
                        stage_id=stage_id,
                        status="failed",
                        error=str(exc),
                    )
                )
                continue

            stage_outputs[stage_id] = dict(output.data)
            frame_updates_by_stage[stage_id] = dict(output.frame_updates)
            context.stage_outputs[stage_id] = dict(output.data)
            stage_records.append(StageExecutionRecord(stage_id=stage_id, status="success"))

        return PipelineExecutionResult(
            stage_outputs=stage_outputs,
            frame_updates_by_stage=frame_updates_by_stage,
            stage_records=stage_records,
            failure=failure,
        )
