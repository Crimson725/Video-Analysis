"""LLM scene packet builder for the KG enrichment pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.scene_bundle import DerivedFrameFeatures, FrameData, SceneBundle

# Default allowed predicates for scene graph relations
DEFAULT_ALLOWED_PREDICATES: list[str] = [
    "appears_in",
    "near",
    "holding",
    "looking_at",
    "speaking_to",
    "wearing",
    "sitting_on",
    "standing_on",
    "inside",
    "next_to",
    "behind",
    "in_front_of",
    "interacting_with",
    "using",
    "carrying",
    "touching",
    "riding",
    "driving",
    "walking_with",
    "handing_to",
    "receiving_from",
    "pushing",
    "pulling",
    "pointing_at",
    "watching",
    "following",
    "approaching",
    "leaving",
    "entering",
    "exiting",
]


class LLMPacketConstraints(BaseModel):
    """Constraints block included in the LLM scene packet."""

    allowed_predicates: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ALLOWED_PREDICATES)
    )
    evidence_required: bool = True
    do_not_invent_coordinates: bool = True
    do_not_override_cv_ids: bool = True


class LLMScenePacket(BaseModel):
    """JSON structure sent to the LLM for scene graph extraction."""

    job_id: str
    scene_id: str
    scene_time_span_s: tuple[float, float]
    source_key: str
    frames: list[dict[str, Any]] = Field(default_factory=list)
    tracks_index: dict[str, dict[str, Any]] = Field(default_factory=dict)
    derived_features: list[dict[str, Any]] = Field(default_factory=list)
    constraints: LLMPacketConstraints = Field(default_factory=LLMPacketConstraints)


def _serialize_frame_data(frame: FrameData) -> dict[str, Any]:
    """Serialize a FrameData into the packet frame schema."""
    tracks_list: list[dict[str, Any]] = []
    for det in frame.tracks:
        track_entry: dict[str, Any] = {
            "track_id": det.get("track_id", ""),
            "label": det.get("label", ""),
            "confidence": det.get("confidence", 0.0),
        }
        if det.get("box"):
            track_entry["box"] = det["box"]
        if det.get("object_track_id"):
            track_entry["object_track_id"] = det["object_track_id"]
        if det.get("person_track_id"):
            track_entry["person_track_id"] = det["person_track_id"]
        if det.get("person_identity_id"):
            track_entry["person_identity_id"] = det["person_identity_id"]
        tracks_list.append(track_entry)

    face_crops: list[dict[str, Any]] = []
    for idx, face in enumerate(frame.faces):
        face_entry: dict[str, Any] = {
            "face_id": face.get("face_id", idx),
            "identity_id": face.get("identity_id", ""),
            "confidence": face.get("confidence", 0.0),
        }
        if face.get("coordinates"):
            face_entry["coordinates"] = face["coordinates"]
        if face.get("video_person_id"):
            face_entry["video_person_id"] = face["video_person_id"]
        if idx < len(frame.face_crop_urls):
            face_entry["crop_url"] = frame.face_crop_urls[idx]
        face_crops.append(face_entry)

    return {
        "frame_id": frame.frame_id,
        "timestamp_sec": frame.timestamp_sec,
        "images": {
            "original_url": frame.original_url,
            "overlay_url": frame.overlay_url,
            "face_crops": face_crops,
        },
        "tracks": tracks_list,
    }


def _serialize_derived_features(features: DerivedFrameFeatures) -> dict[str, Any]:
    """Serialize DerivedFrameFeatures into the packet schema."""
    return {
        "frame_id": features.frame_id,
        "centroids": {k: list(v) for k, v in features.centroids.items()},
        "bbox_area_pct": dict(features.bbox_area_pct),
        "polygon_area_pct": dict(features.polygon_area_pct),
        "pairwise_distances": list(features.pairwise_distances),
        "pairwise_iou": list(features.pairwise_iou),
        "spatial_ordering": list(features.spatial_ordering),
        "near_edges": list(features.near_edges),
        "velocities": dict(features.velocities),
    }


def _serialize_tracks_index(bundle: SceneBundle) -> dict[str, dict[str, Any]]:
    """Serialize tracks_index to JSON-ready dict."""
    result: dict[str, dict[str, Any]] = {}
    for tid, meta in bundle.tracks_index.items():
        result[tid] = {
            "track_id": meta.track_id,
            "label": meta.label,
            "entity_type": meta.entity_type,
            "confidence_mean": meta.confidence_mean,
            "first_frame_id": meta.first_frame_id,
            "last_frame_id": meta.last_frame_id,
            "frame_ids": meta.frame_ids,
        }
        if meta.video_person_id:
            result[tid]["video_person_id"] = meta.video_person_id
    return result


def build_llm_packet(
    bundle: SceneBundle,
    *,
    allowed_predicates: list[str] | None = None,
) -> LLMScenePacket:
    """Transform a SceneBundle into the JSON packet for LLM input."""
    frames_data = [_serialize_frame_data(f) for f in bundle.frames]
    derived = [_serialize_derived_features(f) for f in bundle.derived_features]
    tracks = _serialize_tracks_index(bundle)
    constraints = LLMPacketConstraints(
        allowed_predicates=allowed_predicates or list(DEFAULT_ALLOWED_PREDICATES),
    )

    return LLMScenePacket(
        job_id=bundle.job_id,
        scene_id=bundle.scene_id,
        scene_time_span_s=bundle.scene_time_span,
        source_key=bundle.source_key,
        frames=frames_data,
        tracks_index=tracks,
        derived_features=derived,
        constraints=constraints,
    )
