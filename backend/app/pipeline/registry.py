"""Registry/configuration for modular pipeline stages."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from app.pipeline.contracts import PipelineStage, StageConfig


@dataclass(frozen=True, slots=True)
class RegisteredStage:
    """Registered stage together with effective enablement."""

    stage: PipelineStage
    enabled: bool


def validate_stage_contract(stage: object) -> None:
    """Validate that an object satisfies the required stage contract."""
    stage_id = getattr(stage, "stage_id", None)
    dependencies = getattr(stage, "dependencies", None)
    run = getattr(stage, "run", None)

    if not isinstance(stage_id, str) or not stage_id.strip():
        raise ValueError("Stage contract requires non-empty string 'stage_id'")
    if not isinstance(dependencies, tuple):
        raise ValueError(
            f"Stage '{stage_id}' contract requires tuple 'dependencies'"
        )
    if not all(isinstance(dep, str) and dep for dep in dependencies):
        raise ValueError(
            f"Stage '{stage_id}' contract requires string dependency identifiers"
        )
    if not callable(run):
        raise ValueError(f"Stage '{stage_id}' contract requires callable 'run'")


class PipelineStageRegistry:
    """Registry preserving insertion order and dependency-deterministic planning."""

    def __init__(self) -> None:
        self._stages: OrderedDict[str, PipelineStage] = OrderedDict()
        self._config: dict[str, StageConfig] = {}

    def register(self, stage: PipelineStage, *, enabled: bool = True) -> None:
        """Register a stage and its enablement flag."""
        validate_stage_contract(stage)
        self._stages[stage.stage_id] = stage
        self._config[stage.stage_id] = StageConfig(enabled=enabled)

    def set_enabled(self, stage_id: str, enabled: bool) -> None:
        """Toggle stage enablement by id."""
        if stage_id not in self._stages:
            raise KeyError(f"Unknown stage '{stage_id}'")
        self._config[stage_id] = StageConfig(enabled=enabled)

    def ordered(self) -> list[RegisteredStage]:
        """Return dependency-safe deterministic execution order."""
        ordered_ids = self._topological_stage_order()
        return [
            RegisteredStage(
                stage=self._stages[stage_id],
                enabled=self._config.get(stage_id, StageConfig()).enabled,
            )
            for stage_id in ordered_ids
        ]

    def _topological_stage_order(self) -> list[str]:
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: list[str] = []

        def visit(stage_id: str) -> None:
            if stage_id in visited:
                return
            if stage_id in visiting:
                raise ValueError(f"Cyclic stage dependency detected at '{stage_id}'")
            if stage_id not in self._stages:
                raise ValueError(f"Stage dependency '{stage_id}' is not registered")

            visiting.add(stage_id)
            stage = self._stages[stage_id]
            for dep in stage.dependencies:
                if dep not in self._stages:
                    raise ValueError(
                        f"Stage '{stage.stage_id}' depends on missing stage '{dep}'"
                    )
                visit(dep)
            visiting.remove(stage_id)
            visited.add(stage_id)
            ordered.append(stage_id)

        for stage_id in self._stages:
            visit(stage_id)
        return ordered
