"""Scene-level narrative and full-video synopsis pipeline."""

from __future__ import annotations

import json
import logging
import operator
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Protocol, TypedDict

from pydantic import BaseModel, Field, ValidationError

from app.storage import build_scene_key, build_summary_key

if TYPE_CHECKING:
    from app.config import Settings
    from app.storage import MediaStore

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import availability depends on runtime env
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - import availability depends on runtime env
    LANGGRAPH_AVAILABLE = False
    END = START = StateGraph = None  # type: ignore[assignment]


def _parse_timestamp_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm timestamps to seconds."""
    parts = timestamp.split(":")
    if len(parts) != 3:
        return 0.0
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _safe_toon_value(value: str) -> str:
    """Normalize values for compact TOON row encoding."""
    return value.replace(",", " ").replace("\n", " ").strip() or "unknown"


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


@dataclass(slots=True)
class ScenePacket:
    """TOON-first scene packet used as narrative model input."""

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
    toon_payload: str
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
        """Generate structured scene narrative from a TOON prompt."""

    def refine_synopsis(
        self,
        current_synopsis: str,
        scene_narrative: SceneNarrative,
    ) -> str:
        """Refine or initialize synopsis with one scene narrative."""


class FallbackSceneLLMClient:
    """Deterministic no-network fallback used when Gemini runtime is unavailable."""

    def __init__(self, scene_model_id: str, synopsis_model_id: str) -> None:
        self.scene_model_id = scene_model_id
        self.synopsis_model_id = synopsis_model_id

    def generate_scene_narrative(self, prompt: str, scene_packet: ScenePacket) -> dict[str, Any]:
        del prompt
        if scene_packet.entities:
            top_entities = ", ".join(e.name for e in scene_packet.entities[:3])
        else:
            top_entities = "no dominant entities"
        paragraph = (
            f"From {scene_packet.start_sec:.2f}s to {scene_packet.end_sec:.2f}s, "
            f"the scene shows {top_entities} with {scene_packet.objects_total} object signals "
            f"and {scene_packet.faces_total} face detections."
        )
        moments = [
            f"Scene duration is {scene_packet.duration_sec:.2f} seconds.",
            f"Detected {scene_packet.unique_labels} unique labels.",
        ]
        if scene_packet.events:
            moments.append(f"Notable event: {scene_packet.events[0].event}.")
        return {
            "narrative_paragraph": paragraph,
            "key_moments": moments,
            "mentioned_entities": [e.name for e in scene_packet.entities[:3]],
            "mentioned_events": [e.event for e in scene_packet.events[:3]],
        }

    def refine_synopsis(
        self,
        current_synopsis: str,
        scene_narrative: SceneNarrative,
    ) -> str:
        chunk = (
            f"[{scene_narrative.start_sec:.2f}-{scene_narrative.end_sec:.2f}s] "
            f"{scene_narrative.narrative_paragraph}"
        )
        if not current_synopsis:
            return chunk
        return f"{current_synopsis} {chunk}"


class GeminiSceneLLMClient:
    """Gemini-backed scene and synopsis generation client."""

    def __init__(self, google_api_key: str, scene_model_id: str, synopsis_model_id: str) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._scene_model = ChatGoogleGenerativeAI(
            model=scene_model_id,
            google_api_key=google_api_key,
            temperature=0.1,
        )
        self._synopsis_model = ChatGoogleGenerativeAI(
            model=synopsis_model_id,
            google_api_key=google_api_key,
            temperature=0.1,
        )

    @staticmethod
    def _to_text(response: Any) -> str:
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                blocks: list[str] = []
                for block in content:
                    if isinstance(block, str):
                        blocks.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        blocks.append(str(block["text"]))
                return "\n".join(blocks)
        return str(response)

    @staticmethod
    def _parse_scene_json(text: str) -> dict[str, Any]:
        """Parse scene JSON from raw/fenced/wrapped model output."""
        candidates: list[str] = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        fenced_blocks = re.findall(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            stripped,
            flags=re.IGNORECASE,
        )
        candidates.extend(block.strip() for block in fenced_blocks if block.strip())

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidates.append(stripped[first_brace : last_brace + 1].strip())

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("Gemini scene narrative response is not a JSON object")

    def generate_scene_narrative(self, prompt: str, scene_packet: ScenePacket) -> dict[str, Any]:
        del scene_packet
        response = self._scene_model.invoke(prompt)
        text = self._to_text(response).strip()
        return self._parse_scene_json(text)

    def refine_synopsis(
        self,
        current_synopsis: str,
        scene_narrative: SceneNarrative,
    ) -> str:
        if current_synopsis:
            prompt = (
                "Refine the existing video synopsis while preserving chronology and voice.\n"
                f"Current synopsis:\n{current_synopsis}\n\n"
                f"Next scene ({scene_narrative.start_sec:.2f}-{scene_narrative.end_sec:.2f}s):\n"
                f"{scene_narrative.narrative_paragraph}\n"
                f"Key moments: {json.dumps(scene_narrative.key_moments)}\n\n"
                "Return only the updated synopsis paragraph."
            )
        else:
            prompt = (
                "Create an initial video synopsis paragraph from this first scene.\n"
                f"Scene ({scene_narrative.start_sec:.2f}-{scene_narrative.end_sec:.2f}s):\n"
                f"{scene_narrative.narrative_paragraph}\n"
                f"Key moments: {json.dumps(scene_narrative.key_moments)}\n\n"
                "Return only the synopsis paragraph."
            )
        response = self._synopsis_model.invoke(prompt)
        return self._to_text(response).strip()


def build_scene_llm_client(settings: "Settings") -> SceneLLMClient:
    """Build configured scene LLM client with graceful fallback behavior."""
    if settings.google_api_key:
        try:
            return GeminiSceneLLMClient(
                google_api_key=settings.google_api_key,
                scene_model_id=settings.scene_model_id,
                synopsis_model_id=settings.synopsis_model_id,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency path
            logger.warning("Gemini client unavailable; using fallback client: %s", exc)
    return FallbackSceneLLMClient(
        scene_model_id=settings.scene_model_id,
        synopsis_model_id=settings.synopsis_model_id,
    )


def _trace_metadata(
    settings: "Settings",
    *,
    stage: str,
    job_id: str,
    scene_id: int | None = None,
    model: str | None = None,
) -> dict[str, str] | None:
    """Build trace metadata payload when tracing is enabled."""
    if not settings.langsmith_tracing_enabled:
        return None
    metadata: dict[str, str] = {
        "stage": stage,
        "job_id": job_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    if settings.langsmith_project:
        metadata["project"] = settings.langsmith_project
    if scene_id is not None:
        metadata["scene_id"] = str(scene_id)
    if model:
        metadata["model"] = model
    return metadata


def _select_scene_frames(
    frame_results: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame in frame_results:
        ts = _parse_timestamp_seconds(str(frame.get("timestamp", "")))
        if start_sec <= ts <= end_sec:
            selected.append(frame)
    return selected


def _collect_entities(scene_frames: list[dict[str, Any]], settings: "Settings") -> list[EntityCount]:
    counts: dict[tuple[str, str], int] = {}
    for frame in scene_frames:
        analysis = frame.get("analysis", {})
        for det in analysis.get("object_detection", []):
            name = _safe_toon_value(str(det.get("label", "unknown")))
            key = (name, "object")
            counts[key] = counts.get(key, 0) + 1
        for seg in analysis.get("semantic_segmentation", []):
            name = _safe_toon_value(str(seg.get("class", "unknown")))
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


def _collect_events(
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


def _select_keyframes(
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
        lines.append(f"  {_safe_toon_value(entity.name)},{entity.entity_type},{entity.count}")

    lines.append(f"events[{len(packet.events)}]{{event,count,evidence}}:")
    for event in packet.events:
        lines.append(
            f"  {_safe_toon_value(event.event)},{event.count},{_safe_toon_value(event.evidence)}"
        )

    lines.append(f"keyframes[{len(packet.keyframes)}]{{frameId,timestamp,uri}}:")
    for keyframe in packet.keyframes:
        lines.append(
            f"  {keyframe.frame_id},{_safe_toon_value(keyframe.timestamp)},{_safe_toon_value(keyframe.uri)}"
        )
    return "\n".join(lines) + "\n"


def build_scene_prompt(scene_packet: ScenePacket) -> str:
    """Build TOON-first prompt for one-call-per-scene narrative generation."""
    keyframe_note = "No keyframes attached."
    if scene_packet.keyframes:
        refs = ", ".join(k.uri for k in scene_packet.keyframes)
        keyframe_note = f"Supporting keyframes: {refs}"
    return (
        "You are given one scene packet in TOON format. "
        "Generate a faithful narrative for only this scene.\n\n"
        "Rules:\n"
        "- Use only entities/events present in the packet.\n"
        "- Return JSON with keys: narrative_paragraph, key_moments, mentioned_entities, mentioned_events.\n"
        "- key_moments must be a non-empty array of concise bullet strings.\n"
        "- Keep narrative_paragraph short and chronological.\n\n"
        f"{keyframe_note}\n\n"
        "```toon\n"
        f"{scene_packet.toon_payload}"
        "```\n"
    )


def _validate_faithfulness(scene_packet: ScenePacket, narrative: SceneNarrativeModel) -> None:
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


def sort_scene_narratives(scene_narratives: list[SceneNarrative]) -> list[SceneNarrative]:
    """Sort scene narratives by chronological scene start time."""
    return sorted(scene_narratives, key=lambda item: item.start_sec)


def _build_scene_packets(
    job_id: str,
    scenes: list[tuple[float, float]],
    frame_results: list[dict[str, Any]],
    settings: "Settings",
    media_store: "MediaStore",
) -> list[ScenePacket]:
    packets: list[ScenePacket] = []
    for scene_id, (start_sec, end_sec) in enumerate(scenes):
        scene_frames = _select_scene_frames(frame_results, start_sec, end_sec)
        entities = _collect_entities(scene_frames, settings)
        objects_total = sum(item.count for item in entities if item.entity_type == "object")
        faces_total = sum(item.count for item in entities if item.name == "face")
        unique_labels = len({item.name for item in entities})
        keyframes = _select_keyframes(
            scene_frames,
            unique_labels=unique_labels,
            settings=settings,
        )
        events = _collect_events(entities, faces_total, settings)

        packet_key = build_scene_key(job_id, "packet", scene_id)
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


def _generate_narratives(
    *,
    job_id: str,
    scene_packets: list[ScenePacket],
    llm_client: SceneLLMClient,
    settings: "Settings",
    media_store: "MediaStore",
) -> list[SceneNarrative]:
    narratives: list[SceneNarrative] = []
    for packet in scene_packets:
        prompt = build_scene_prompt(packet)
        last_error: Exception | None = None
        parsed: SceneNarrativeModel | None = None
        for _attempt in range(settings.scene_llm_retry_count + 1):
            try:
                raw = llm_client.generate_scene_narrative(prompt, packet)
                parsed = SceneNarrativeModel.model_validate(raw)
                _validate_faithfulness(packet, parsed)
                break
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                continue
        if parsed is None:
            raise RuntimeError(
                f"Scene narrative generation failed for scene {packet.scene_id}"
            ) from last_error

        narrative_key = build_scene_key(job_id, "narrative", packet.scene_id)
        narrative_payload = {
            "scene_id": packet.scene_id,
            "start_sec": packet.start_sec,
            "end_sec": packet.end_sec,
            "narrative_paragraph": parsed.narrative_paragraph,
            "key_moments": parsed.key_moments,
            "packet": packet.packet_key,
        }
        media_store.upload_scene_artifact(
            job_id=job_id,
            artifact_kind="narrative",
            scene_id=packet.scene_id,
            payload=json.dumps(narrative_payload, separators=(",", ":")).encode("utf-8"),
        )
        narratives.append(
            SceneNarrative(
                scene_id=packet.scene_id,
                start_sec=packet.start_sec,
                end_sec=packet.end_sec,
                narrative_paragraph=parsed.narrative_paragraph,
                key_moments=parsed.key_moments,
                packet_key=packet.packet_key,
                narrative_key=narrative_key,
                trace=_trace_metadata(
                    settings,
                    stage="scene_narrative",
                    job_id=job_id,
                    scene_id=packet.scene_id,
                    model=settings.scene_model_id,
                ),
            )
        )
    return narratives


def _build_synopsis(
    *,
    job_id: str,
    scene_narratives: list[SceneNarrative],
    llm_client: SceneLLMClient,
    settings: "Settings",
    media_store: "MediaStore",
) -> dict[str, Any] | None:
    ordered = sort_scene_narratives(scene_narratives)
    if not ordered:
        return None
    synopsis = ""
    for narrative in ordered:
        synopsis = llm_client.refine_synopsis(synopsis, narrative).strip()
    summary_key = build_summary_key(job_id, "synopsis")
    payload = {
        "synopsis": synopsis,
        "model": settings.synopsis_model_id,
        "scene_count": len(ordered),
    }
    media_store.upload_summary_artifact(
        job_id=job_id,
        artifact_kind="synopsis",
        payload=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )
    return {
        "synopsis": synopsis,
        "artifact": summary_key,
        "model": settings.synopsis_model_id,
        "trace": _trace_metadata(
            settings,
            stage="video_synopsis",
            job_id=job_id,
            model=settings.synopsis_model_id,
        ),
    }


class WorkflowState(TypedDict):
    """LangGraph workflow state."""

    job_id: str
    scenes: list[tuple[float, float]]
    frame_results: list[dict[str, Any]]
    scene_packets: Annotated[list[ScenePacket], operator.add]
    scene_narratives: Annotated[list[SceneNarrative], operator.add]
    synopsis: str


def _run_with_langgraph(
    *,
    job_id: str,
    scenes: list[tuple[float, float]],
    frame_results: list[dict[str, Any]],
    settings: "Settings",
    media_store: "MediaStore",
    llm_client: SceneLLMClient,
) -> dict[str, Any]:
    """Execute workflow via LangGraph when available."""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph is not available")

    def packets_node(state: WorkflowState) -> dict[str, Any]:
        packets = _build_scene_packets(
            job_id=state["job_id"],
            scenes=state["scenes"],
            frame_results=state["frame_results"],
            settings=settings,
            media_store=media_store,
        )
        return {"scene_packets": packets}

    def narratives_node(state: WorkflowState) -> dict[str, Any]:
        narratives = _generate_narratives(
            job_id=state["job_id"],
            scene_packets=state["scene_packets"],
            llm_client=llm_client,
            settings=settings,
            media_store=media_store,
        )
        return {"scene_narratives": narratives}

    def synopsis_node(state: WorkflowState) -> dict[str, Any]:
        synopsis_payload = _build_synopsis(
            job_id=state["job_id"],
            scene_narratives=state["scene_narratives"],
            llm_client=llm_client,
            settings=settings,
            media_store=media_store,
        )
        text = ""
        if synopsis_payload is not None:
            text = synopsis_payload["synopsis"]
        return {"synopsis": text}

    graph_builder = StateGraph(WorkflowState)
    graph_builder.add_node("packets", packets_node)
    graph_builder.add_node("narratives", narratives_node)
    graph_builder.add_node("synopsis", synopsis_node)
    graph_builder.add_edge(START, "packets")
    graph_builder.add_edge("packets", "narratives")
    graph_builder.add_edge("narratives", "synopsis")
    graph_builder.add_edge("synopsis", END)
    graph = graph_builder.compile()
    output = graph.invoke(
        {
            "job_id": job_id,
            "scenes": scenes,
            "frame_results": frame_results,
            "scene_packets": [],
            "scene_narratives": [],
            "synopsis": "",
        }
    )
    narratives = sort_scene_narratives(output.get("scene_narratives", []))
    synopsis_payload = _build_synopsis(
        job_id=job_id,
        scene_narratives=narratives,
        llm_client=llm_client,
        settings=settings,
        media_store=media_store,
    )
    return {
        "scene_narratives": [
            {
                "scene_id": item.scene_id,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "narrative_paragraph": item.narrative_paragraph,
                "key_moments": item.key_moments,
                "artifacts": {"packet": item.packet_key, "narrative": item.narrative_key},
                "trace": item.trace,
            }
            for item in narratives
        ],
        "video_synopsis": synopsis_payload,
    }


def run_scene_understanding_pipeline(
    *,
    job_id: str,
    scenes: list[tuple[float, float]],
    frame_results: list[dict[str, Any]],
    settings: "Settings",
    media_store: "MediaStore",
) -> dict[str, Any]:
    """Generate scene packets, scene narratives, and final video synopsis."""
    if not scenes:
        return {"scene_narratives": [], "video_synopsis": None}

    llm_client = build_scene_llm_client(settings)
    if LANGGRAPH_AVAILABLE:
        try:
            return _run_with_langgraph(
                job_id=job_id,
                scenes=scenes,
                frame_results=frame_results,
                settings=settings,
                media_store=media_store,
                llm_client=llm_client,
            )
        except Exception as exc:
            logger.warning("LangGraph execution failed; falling back to sequential run: %s", exc)

    scene_packets = _build_scene_packets(
        job_id=job_id,
        scenes=scenes,
        frame_results=frame_results,
        settings=settings,
        media_store=media_store,
    )
    scene_narratives = _generate_narratives(
        job_id=job_id,
        scene_packets=scene_packets,
        llm_client=llm_client,
        settings=settings,
        media_store=media_store,
    )
    video_synopsis = _build_synopsis(
        job_id=job_id,
        scene_narratives=scene_narratives,
        llm_client=llm_client,
        settings=settings,
        media_store=media_store,
    )
    return {
        "scene_narratives": [
            {
                "scene_id": item.scene_id,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "narrative_paragraph": item.narrative_paragraph,
                "key_moments": item.key_moments,
                "artifacts": {"packet": item.packet_key, "narrative": item.narrative_key},
                "trace": item.trace,
            }
            for item in sort_scene_narratives(scene_narratives)
        ],
        "video_synopsis": video_synopsis,
    }
