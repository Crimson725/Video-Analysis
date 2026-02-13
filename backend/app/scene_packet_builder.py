"""Deterministic scene packet construction and contract validation utilities."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from app.scene_runtime_contracts import (
    EntityCount,
    EventCount,
    KeyframeRef,
    SceneNarrativeModel,
    ScenePacket,
)
from app.storage import build_corpus_key, build_scene_key

if TYPE_CHECKING:
    from app.config import Settings
    from app.storage import MediaStore


def parse_timestamp_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm timestamps to seconds."""
    parts = timestamp.split(":")
    if len(parts) != 3:
        return 0.0
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def safe_toon_value(value: str) -> str:
    """Normalize values for compact TOON row encoding."""
    return value.replace(",", " ").replace("\n", " ").strip() or "unknown"


def select_scene_frames(
    frame_results: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame in frame_results:
        ts = parse_timestamp_seconds(str(frame.get("timestamp", "")))
        if start_sec <= ts <= end_sec:
            selected.append(frame)
    return selected


def collect_entities(scene_frames: list[dict[str, Any]], settings: "Settings") -> list[EntityCount]:
    counts: dict[tuple[str, str], int] = {}
    for frame in scene_frames:
        analysis = frame.get("analysis", {})
        for det in analysis.get("object_detection", []):
            name = safe_toon_value(str(det.get("label", "unknown")))
            key = (name, "object")
            counts[key] = counts.get(key, 0) + 1
        for seg in analysis.get("semantic_segmentation", []):
            name = safe_toon_value(str(seg.get("class", "unknown")))
            key = (name, "object")
            counts[key] = counts.get(key, 0) + 1
        face_count = len(analysis.get("face_recognition", []))
        if face_count > 0:
            key = ("face", "human")
            counts[key] = counts.get(key, 0) + face_count

    entities = [
        EntityCount(name=name, entity_type=entity_type, count=count)
        for (name, entity_type), count in counts.items()
    ]
    entities.sort(key=lambda item: (-item.count, item.name))
    return entities[: settings.scene_packet_max_entities]


def collect_events(
    entities: list[EntityCount],
    faces_total: int,
    settings: "Settings",
) -> list[EventCount]:
    events: list[EventCount] = []
    for entity in entities:
        events.append(
            EventCount(
                event=f"{entity.name}_observed",
                count=entity.count,
                evidence="aggregated_frame_analysis",
            )
        )
    if faces_total > 0:
        events.append(
            EventCount(
                event="face_detected",
                count=faces_total,
                evidence="face_recognition",
            )
        )
    events.sort(key=lambda item: (-item.count, item.event))
    return events[: settings.scene_packet_max_events]


def select_keyframes(
    scene_frames: list[dict[str, Any]],
    *,
    unique_labels: int,
    settings: "Settings",
) -> list[KeyframeRef]:
    should_attach = unique_labels >= settings.scene_packet_disambiguation_label_threshold
    if not should_attach or not scene_frames:
        return []

    max_frames = min(settings.scene_packet_max_keyframes, len(scene_frames))
    if max_frames <= 0:
        return []
    selected = scene_frames[:max_frames]
    refs: list[KeyframeRef] = []
    for frame in selected:
        files = frame.get("files", {})
        refs.append(
            KeyframeRef(
                frame_id=int(frame.get("frame_id", 0)),
                timestamp=str(frame.get("timestamp", "")),
                uri=str(files.get("original", "")),
            )
        )
    return refs


def deterministic_id(prefix: str, *parts: Any) -> str:
    raw = "::".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def build_evidence_anchor(frame: dict[str, Any], *, bbox: list[int] | None, text_span: str) -> dict[str, Any]:
    analysis_artifacts = frame.get("analysis_artifacts", {})
    return {
        "frame_id": int(frame.get("frame_id", 0)),
        "timestamp": str(frame.get("timestamp", "")),
        "artifact_key": str(analysis_artifacts.get("json", "")),
        "bbox": bbox,
        "text_span": text_span,
    }


def collect_corpus_entities(
    *,
    job_id: str,
    scene_id: int,
    scene_frames: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> tuple[list[dict[str, Any]], dict[int, list[str]]]:
    """Aggregate scene entities with temporal spans and evidence anchors."""
    accum: dict[tuple[str, str], dict[str, Any]] = {}
    frame_entity_ids: dict[int, list[str]] = {}

    for frame in scene_frames:
        frame_id = int(frame.get("frame_id", 0))
        ts = parse_timestamp_seconds(str(frame.get("timestamp", "")))
        analysis = frame.get("analysis", {})

        for det in analysis.get("object_detection", []):
            label = str(det.get("label", "unknown"))
            track_id = str(det.get("track_id", "")) or deterministic_id(
                "track",
                job_id,
                scene_id,
                frame_id,
                label,
            )
            key = ("object", track_id)
            entity_id = deterministic_id("entity", job_id, scene_id, key[0], key[1], label)
            evidence = build_evidence_anchor(frame, bbox=det.get("box"), text_span=label)
            slot = accum.setdefault(
                key,
                {
                    "entity_id": entity_id,
                    "label": label,
                    "entity_type": "object",
                    "count": 0,
                    "confidence_sum": 0.0,
                    "first_seen": ts,
                    "last_seen": ts,
                    "track_id": track_id,
                    "identity_id": None,
                    "evidence": [],
                },
            )
            slot["count"] += 1
            slot["confidence_sum"] += float(det.get("confidence", 0.0))
            slot["first_seen"] = min(float(slot["first_seen"]), ts)
            slot["last_seen"] = max(float(slot["last_seen"]), ts)
            slot["evidence"].append(evidence)
            frame_entity_ids.setdefault(frame_id, []).append(entity_id)

        for face in analysis.get("face_recognition", []):
            identity_id = str(face.get("identity_id", "")) or deterministic_id(
                "identity",
                job_id,
                scene_id,
                frame_id,
                face.get("face_id"),
            )
            key = ("person", identity_id)
            entity_id = deterministic_id("entity", job_id, scene_id, key[0], key[1], "person")
            evidence = build_evidence_anchor(
                frame,
                bbox=face.get("coordinates"),
                text_span=identity_id,
            )
            slot = accum.setdefault(
                key,
                {
                    "entity_id": entity_id,
                    "label": identity_id,
                    "entity_type": "person",
                    "count": 0,
                    "confidence_sum": 0.0,
                    "first_seen": ts,
                    "last_seen": ts,
                    "track_id": None,
                    "identity_id": identity_id,
                    "evidence": [],
                },
            )
            slot["count"] += 1
            slot["confidence_sum"] += float(face.get("confidence", 0.0))
            slot["first_seen"] = min(float(slot["first_seen"]), ts)
            slot["last_seen"] = max(float(slot["last_seen"]), ts)
            slot["evidence"].append(evidence)
            frame_entity_ids.setdefault(frame_id, []).append(entity_id)

    entities: list[dict[str, Any]] = []
    for slot in accum.values():
        count = max(1, int(slot["count"]))
        first_seen = max(start_sec, float(slot["first_seen"]))
        last_seen = min(end_sec, float(slot["last_seen"]))
        entities.append(
            {
                "entity_id": slot["entity_id"],
                "label": slot["label"],
                "entity_type": slot["entity_type"],
                "count": count,
                "confidence": float(slot["confidence_sum"]) / count,
                "temporal_span": {
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "duration_sec": max(0.0, last_seen - first_seen),
                },
                "evidence": slot["evidence"][:5],
                "track_id": slot["track_id"],
                "identity_id": slot["identity_id"],
            }
        )

    entities.sort(key=lambda item: (-int(item["count"]), str(item["label"])))
    return entities, frame_entity_ids


def collect_corpus_events(
    *,
    job_id: str,
    scene_id: int,
    entities: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entity in entities:
        event_type = f"{entity['entity_type']}_observed"
        events.append(
            {
                "event_id": deterministic_id(
                    "event",
                    job_id,
                    scene_id,
                    event_type,
                    entity["entity_id"],
                ),
                "event_type": event_type,
                "count": int(entity["count"]),
                "confidence": float(entity["confidence"]),
                "temporal_span": {
                    "first_seen": max(start_sec, float(entity["temporal_span"]["first_seen"])),
                    "last_seen": min(end_sec, float(entity["temporal_span"]["last_seen"])),
                    "duration_sec": float(entity["temporal_span"]["duration_sec"]),
                },
                "evidence": list(entity["evidence"])[:3],
            }
        )
    return events


def collect_corpus_relations(
    *,
    job_id: str,
    scene_id: int,
    scene_frames: list[dict[str, Any]],
    frame_entity_ids: dict[int, list[str]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    relation_counts: dict[tuple[str, str], dict[str, Any]] = {}
    for frame in scene_frames:
        frame_id = int(frame.get("frame_id", 0))
        entity_ids = sorted(set(frame_entity_ids.get(frame_id, [])))
        if len(entity_ids) < 2:
            continue
        ts = parse_timestamp_seconds(str(frame.get("timestamp", "")))
        for index, source_id in enumerate(entity_ids):
            for target_id in entity_ids[index + 1 :]:
                key = (source_id, target_id)
                slot = relation_counts.setdefault(
                    key,
                    {
                        "count": 0,
                        "first_seen": ts,
                        "last_seen": ts,
                        "evidence": [],
                    },
                )
                slot["count"] += 1
                slot["first_seen"] = min(float(slot["first_seen"]), ts)
                slot["last_seen"] = max(float(slot["last_seen"]), ts)
                slot["evidence"].append(
                    build_evidence_anchor(frame, bbox=None, text_span="co_occurs_with")
                )

    relations: list[dict[str, Any]] = []
    for (source_id, target_id), slot in relation_counts.items():
        first_seen = max(start_sec, float(slot["first_seen"]))
        last_seen = min(end_sec, float(slot["last_seen"]))
        relations.append(
            {
                "relation_id": deterministic_id(
                    "relation",
                    job_id,
                    scene_id,
                    source_id,
                    "co_occurs_with",
                    target_id,
                ),
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "predicate": "co_occurs_with",
                "confidence": min(1.0, 0.2 + 0.2 * int(slot["count"])),
                "temporal_span": {
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "duration_sec": max(0.0, last_seen - first_seen),
                },
                "evidence": slot["evidence"][:5],
            }
        )
    return relations


def build_retrieval_chunks(
    *,
    job_id: str,
    scene_id: int,
    start_sec: float,
    end_sec: float,
    entities: list[dict[str, Any]],
    events: list[dict[str, Any]],
    packet_key: str,
) -> list[dict[str, Any]]:
    entity_text = ", ".join(str(entity["label"]) for entity in entities[:6]) or "none"
    event_text = ", ".join(str(event["event_type"]) for event in events[:6]) or "none"
    text = (
        f"Scene {scene_id} ({start_sec:.2f}s-{end_sec:.2f}s) entities: {entity_text}. "
        f"Events: {event_text}."
    )
    return [
        {
            "chunk_id": deterministic_id("chunk", job_id, scene_id, text),
            "text": text,
            "source_entity_ids": [str(entity["entity_id"]) for entity in entities[:8]],
            "artifact_keys": [packet_key],
            "temporal_span": {
                "first_seen": start_sec,
                "last_seen": end_sec,
                "duration_sec": max(0.0, end_sec - start_sec),
            },
        }
    ]


def serialize_scene_packet_toon(packet: ScenePacket) -> str:
    """Serialize scene packet with deterministic compact TOON ordering."""
    lines = [
        "scene{sceneId,startSec,endSec,durationSec,objectsTotal,facesTotal,uniqueLabels,keyframes}:",
        (
            f"  {packet.scene_id},{packet.start_sec:.3f},{packet.end_sec:.3f},"
            f"{packet.duration_sec:.3f},{packet.objects_total},{packet.faces_total},"
            f"{packet.unique_labels},{len(packet.keyframes)}"
        ),
        f"entities[{len(packet.entities)}]{{name,type,count}}:",
    ]
    for entity in packet.entities:
        lines.append(f"  {safe_toon_value(entity.name)},{entity.entity_type},{entity.count}")

    lines.append(f"events[{len(packet.events)}]{{event,count,evidence}}:")
    for event in packet.events:
        lines.append(
            f"  {safe_toon_value(event.event)},{event.count},{safe_toon_value(event.evidence)}"
        )

    lines.append(f"keyframes[{len(packet.keyframes)}]{{frameId,timestamp,uri}}:")
    for keyframe in packet.keyframes:
        lines.append(
            f"  {keyframe.frame_id},{safe_toon_value(keyframe.timestamp)},{safe_toon_value(keyframe.uri)}"
        )
    return "\n".join(lines) + "\n"


def validate_narrative_faithfulness(scene_packet: ScenePacket, narrative: SceneNarrativeModel) -> None:
    """Reject narrative payloads that claim unsupported entities/events."""
    allowed_entities = {item.name for item in scene_packet.entities}
    allowed_events = {item.event for item in scene_packet.events}
    unsupported_entities = [
        entity for entity in narrative.mentioned_entities if entity not in allowed_entities
    ]
    unsupported_events = [
        event for event in narrative.mentioned_events if event not in allowed_events
    ]
    if unsupported_entities or unsupported_events:
        raise ValueError(
            "Narrative faithfulness validation failed. "
            f"Unsupported entities={unsupported_entities}, events={unsupported_events}"
        )


def build_scene_packets(
    *,
    job_id: str,
    scenes: list[tuple[float, float]],
    frame_results: list[dict[str, Any]],
    settings: "Settings",
    media_store: "MediaStore",
) -> list[ScenePacket]:
    packets: list[ScenePacket] = []
    for scene_id, (start_sec, end_sec) in enumerate(scenes):
        scene_frames = select_scene_frames(frame_results, start_sec, end_sec)
        entities = collect_entities(scene_frames, settings)
        objects_total = sum(item.count for item in entities if item.entity_type == "object")
        faces_total = sum(item.count for item in entities if item.name == "face")
        unique_labels = len({item.name for item in entities})
        keyframes = select_keyframes(
            scene_frames,
            unique_labels=unique_labels,
            settings=settings,
        )
        events = collect_events(entities, faces_total, settings)

        packet_key = build_scene_key(job_id, "packet", scene_id)
        graph_bundle_key = build_corpus_key(job_id, "graph", f"scene_{scene_id}.json")
        retrieval_bundle_key = build_corpus_key(job_id, "retrieval", f"scene_{scene_id}.json")
        corpus_entities, frame_entity_ids = collect_corpus_entities(
            job_id=job_id,
            scene_id=scene_id,
            scene_frames=scene_frames,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
        corpus_events = collect_corpus_events(
            job_id=job_id,
            scene_id=scene_id,
            entities=corpus_entities,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
        corpus_relations = collect_corpus_relations(
            job_id=job_id,
            scene_id=scene_id,
            scene_frames=scene_frames,
            frame_entity_ids=frame_entity_ids,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
        retrieval_chunks = build_retrieval_chunks(
            job_id=job_id,
            scene_id=scene_id,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            entities=corpus_entities,
            events=corpus_events,
            packet_key=packet_key,
        )
        packet = ScenePacket(
            scene_id=scene_id,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            duration_sec=max(0.0, float(end_sec) - float(start_sec)),
            objects_total=objects_total,
            faces_total=faces_total,
            unique_labels=unique_labels,
            entities=entities,
            events=events,
            keyframes=keyframes,
            corpus_entities=corpus_entities,
            corpus_events=corpus_events,
            corpus_relations=corpus_relations,
            retrieval_chunks=retrieval_chunks,
            graph_bundle_key=graph_bundle_key,
            retrieval_bundle_key=retrieval_bundle_key,
            toon_payload="",
            packet_key=packet_key,
        )
        toon_payload = serialize_scene_packet_toon(packet)
        packet.toon_payload = toon_payload
        media_store.upload_scene_artifact(
            job_id=job_id,
            artifact_kind="packet",
            scene_id=scene_id,
            payload=toon_payload.encode("utf-8"),
        )
        packets.append(packet)
    return packets
