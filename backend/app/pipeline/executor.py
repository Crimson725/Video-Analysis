"""Deterministic modular pipeline executor."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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

    @staticmethod
    def _resolve_max_workers(settings: object) -> int:
        enabled = bool(getattr(settings, "enable_pipeline_branch_concurrency", False))
        if not enabled:
            return 1
        frame_budget = max(
            1,
            int(getattr(settings, "pipeline_frame_branch_worker_budget", 1)),
        )
        chunk_budget = max(
            1,
            int(getattr(settings, "pipeline_chunk_branch_worker_budget", 1)),
        )
        # The executor runs stage-level tasks, so a small fixed cap is sufficient.
        return max(1, min(8, frame_budget + chunk_budget))

    def _execute_sequential(
        self,
        *,
        context: PipelineContext,
        registry: PipelineStageRegistry,
        on_stage_started: StageCallback | None,
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

    def execute(
        self,
        *,
        context: PipelineContext,
        registry: PipelineStageRegistry,
        on_stage_started: StageCallback | None = None,
    ) -> PipelineExecutionResult:
        if self._resolve_max_workers(context.settings) <= 1:
            return self._execute_sequential(
                context=context,
                registry=registry,
                on_stage_started=on_stage_started,
            )

        ordered = registry.ordered()
        ordered_ids = [registered.stage.stage_id for registered in ordered]
        state_by_stage: dict[str, str] = {stage_id: "pending" for stage_id in ordered_ids}
        stage_outputs: dict[str, dict] = {}
        frame_updates_by_stage: dict[str, dict[int, dict]] = {}
        stage_records_by_stage: dict[str, StageExecutionRecord] = {}
        failed_or_blocked: set[str] = set()
        failure: StageFailure | None = None
        max_workers = self._resolve_max_workers(context.settings)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight: dict[Future[StageExecutionOutput], str] = {}

            while True:
                made_progress = False

                for registered in ordered:
                    stage_id = registered.stage.stage_id
                    if state_by_stage[stage_id] != "pending":
                        continue

                    if not registered.enabled:
                        state_by_stage[stage_id] = "done"
                        stage_records_by_stage[stage_id] = StageExecutionRecord(
                            stage_id=stage_id,
                            status="skipped",
                            skipped_reason="disabled",
                        )
                        made_progress = True
                        continue

                    if any(
                        state_by_stage.get(dep) in {"pending", "running"}
                        for dep in registered.stage.dependencies
                    ):
                        continue

                    if any(dep in failed_or_blocked for dep in registered.stage.dependencies):
                        failed_or_blocked.add(stage_id)
                        state_by_stage[stage_id] = "done"
                        stage_records_by_stage[stage_id] = StageExecutionRecord(
                            stage_id=stage_id,
                            status="skipped",
                            skipped_reason="dependency_failed",
                        )
                        made_progress = True
                        continue

                    if on_stage_started is not None:
                        on_stage_started(stage_id)
                    future = executor.submit(registered.stage.run, context)
                    in_flight[future] = stage_id
                    state_by_stage[stage_id] = "running"
                    made_progress = True

                if not in_flight and all(
                    stage_state == "done" for stage_state in state_by_stage.values()
                ):
                    break

                if not in_flight:
                    if not made_progress:
                        for stage_id, stage_state in state_by_stage.items():
                            if stage_state != "pending":
                                continue
                            failed_or_blocked.add(stage_id)
                            state_by_stage[stage_id] = "done"
                            stage_records_by_stage[stage_id] = StageExecutionRecord(
                                stage_id=stage_id,
                                status="skipped",
                                skipped_reason="dependency_failed",
                            )
                    continue

                completed, _ = wait(
                    tuple(in_flight),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    stage_id = in_flight.pop(future)
                    try:
                        output = future.result()
                    except Exception as exc:  # pragma: no cover - validated in unit tests
                        if failure is None:
                            failure = StageFailure(
                                stage_id=stage_id,
                                error=str(exc),
                                exception_type=type(exc).__name__,
                            )
                        failed_or_blocked.add(stage_id)
                        stage_records_by_stage[stage_id] = StageExecutionRecord(
                            stage_id=stage_id,
                            status="failed",
                            error=str(exc),
                        )
                        state_by_stage[stage_id] = "done"
                        continue

                    stage_outputs[stage_id] = dict(output.data)
                    frame_updates_by_stage[stage_id] = dict(output.frame_updates)
                    context.stage_outputs[stage_id] = dict(output.data)
                    stage_records_by_stage[stage_id] = StageExecutionRecord(
                        stage_id=stage_id, status="success"
                    )
                    state_by_stage[stage_id] = "done"

        stage_records = [
            stage_records_by_stage[stage_id]
            for stage_id in ordered_ids
            if stage_id in stage_records_by_stage
        ]

        return PipelineExecutionResult(
            stage_outputs=stage_outputs,
            frame_updates_by_stage=frame_updates_by_stage,
            stage_records=stage_records,
            failure=failure,
        )
