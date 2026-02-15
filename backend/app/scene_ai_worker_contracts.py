"""Typed contracts for queue-dispatched scene AI worker tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SceneWorkerTaskInput:
    """Task payload persisted in the scene AI queue."""

    job_id: str
    scenes: list[tuple[float, float]]
    frame_results: list[dict[str, Any]]
    source_key: str
    video_face_identities: dict[str, Any] | None = None
    video_person_tracks: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scenes": [[float(start), float(end)] for start, end in self.scenes],
            "frame_results": self.frame_results,
            "source_key": self.source_key,
            "video_face_identities": self.video_face_identities,
            "video_person_tracks": self.video_person_tracks,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SceneWorkerTaskInput":
        raw_scenes = payload.get("scenes", [])
        scenes: list[tuple[float, float]] = []
        if isinstance(raw_scenes, list):
            for item in raw_scenes:
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and isinstance(item[0], (int, float))
                    and isinstance(item[1], (int, float))
                ):
                    scenes.append((float(item[0]), float(item[1])))
        frame_results: list[dict[str, Any]] = []
        raw_frames = payload.get("frame_results", [])
        if isinstance(raw_frames, list):
            for item in raw_frames:
                if isinstance(item, dict):
                    frame_results.append(item)
        return cls(
            job_id=str(payload.get("job_id", "")),
            scenes=scenes,
            frame_results=frame_results,
            source_key=str(payload.get("source_key", "")),
            video_face_identities=(
                payload.get("video_face_identities")
                if isinstance(payload.get("video_face_identities"), dict)
                else None
            ),
            video_person_tracks=(
                payload.get("video_person_tracks")
                if isinstance(payload.get("video_person_tracks"), dict)
                else None
            ),
        )

    def idempotency_key(self) -> str:
        return f"{self.job_id}:scene_understanding:v1"


def build_worker_provenance(
    *,
    worker_id: str,
    attempt: int,
    scene_model_id: str,
    synopsis_model_id: str,
    prompt_version: str,
    runtime_version: str,
) -> dict[str, Any]:
    """Build execution provenance persisted with scene outputs."""
    return {
        "worker_id": worker_id,
        "attempt": attempt,
        "scene_model_id": scene_model_id,
        "synopsis_model_id": synopsis_model_id,
        "prompt_version": prompt_version,
        "runtime_version": runtime_version,
    }


def attach_worker_provenance(
    *,
    scene_outputs: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Attach worker provenance to scene narratives and synopsis traces."""
    outputs = dict(scene_outputs)
    narratives = outputs.get("scene_narratives", [])
    if isinstance(narratives, list):
        updated_narratives: list[dict[str, Any]] = []
        for item in narratives:
            if not isinstance(item, dict):
                continue
            trace = item.get("trace")
            trace_map: dict[str, Any] = dict(trace) if isinstance(trace, dict) else {}
            trace_map["worker"] = provenance
            updated_narratives.append({**item, "trace": trace_map})
        outputs["scene_narratives"] = updated_narratives

    synopsis = outputs.get("video_synopsis")
    if isinstance(synopsis, dict):
        trace = synopsis.get("trace")
        trace_map = dict(trace) if isinstance(trace, dict) else {}
        trace_map["worker"] = provenance
        outputs["video_synopsis"] = {**synopsis, "trace": trace_map}
    return outputs
