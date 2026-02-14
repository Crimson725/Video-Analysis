"""Typed contracts shared across deterministic scene logic and runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


@dataclass(slots=True)
class EntityCount:
    """Aggregate count for a scene-level entity."""

    name: str
    entity_type: str
    count: int


@dataclass(slots=True)
class EventCount:
    """Aggregate count for a scene-level event."""

    event: str
    count: int
    evidence: str


@dataclass(slots=True)
class KeyframeRef:
    """Keyframe reference attached to a scene packet."""

    frame_id: int
    timestamp: str
    uri: str


SceneEvidenceModality = Literal[
    "original",
    "object_detection",
    "semantic_segmentation",
    "face_recognition",
]


@dataclass(slots=True)
class SceneFrameContext:
    """Frame-level context carried into multimodal scene understanding."""

    frame_id: int
    timestamp: str
    json_artifact_key: str
    has_faces: bool
    modality_image_keys: dict[SceneEvidenceModality, str]
    analysis_summary: dict[str, Any]


@dataclass(slots=True)
class SceneEvidenceRef:
    """Deterministic link between one image artifact and one frame JSON payload."""

    evidence_id: str
    scene_id: int
    frame_id: int
    timestamp: str
    modality: SceneEvidenceModality
    image_uri: str
    json_artifact_key: str


@dataclass(slots=True)
class ScenePacket:
    """JSON-first scene packet used as narrative model input."""

    scene_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    objects_total: int
    faces_total: int
    unique_labels: int
    entities: list[EntityCount]
    events: list[EventCount]
    keyframes: list[KeyframeRef]
    scene_frames: list[SceneFrameContext]
    evidence_refs: list[SceneEvidenceRef]
    evidence_index: dict[str, dict[str, Any]]
    corpus_entities: list[dict[str, Any]]
    corpus_events: list[dict[str, Any]]
    corpus_relations: list[dict[str, Any]]
    retrieval_chunks: list[dict[str, Any]]
    graph_bundle_key: str
    retrieval_bundle_key: str
    packet_payload: str
    packet_key: str


@dataclass(slots=True)
class SceneNarrative:
    """Generated narrative for a single scene."""

    scene_id: int
    start_sec: float
    end_sec: float
    narrative_paragraph: str
    key_moments: list[str]
    packet_key: str
    narrative_key: str
    corpus: dict[str, Any] | None
    trace: dict[str, str] | None


class SceneNarrativeModel(BaseModel):
    """Structured model contract for scene narrative outputs."""

    narrative_paragraph: str
    key_moments: list[str] = Field(min_length=1)
    mentioned_entities: list[str] = Field(default_factory=list)
    mentioned_events: list[str] = Field(default_factory=list)


class SceneLLMClient(Protocol):
    """LLM abstraction for scene narrative and synopsis refinement."""

    def generate_scene_narrative(self, prompt: str, scene_packet: ScenePacket) -> dict[str, Any]:
        """Generate structured scene narrative from a JSON packet prompt."""

    def refine_synopsis(
        self,
        current_synopsis: str,
        scene_narrative: SceneNarrative,
    ) -> str:
        """Refine or initialize synopsis with one scene narrative."""
