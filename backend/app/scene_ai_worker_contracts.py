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
    video_object_tracks: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scenes": [[float(start), float(end)] for start, end in self.scenes],
            "frame_results": self.frame_results,
            "source_key": self.source_key,
            "video_face_identities": self.video_face_identities,
            "video_person_tracks": self.video_person_tracks,
            "video_object_tracks": self.video_object_tracks,
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
            video_object_tracks=(
                payload.get("video_object_tracks")
                if isinstance(payload.get("video_object_tracks"), dict)
                else None
            ),
        )

    def idempotency_key(self) -> str:
        return f"{self.job_id}:scene_worker:v1"
