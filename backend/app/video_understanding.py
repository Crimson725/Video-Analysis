"""Scene-level narrative and full-video synopsis pipeline orchestration."""

from __future__ import annotations

import json
import logging
import operator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from app.scene_packet_builder import build_scene_packets, serialize_scene_packet_toon
from app.scene_runtime_contracts import SceneLLMClient, SceneNarrative, ScenePacket
from app.scene_worker_runtime import (
    FallbackSceneLLMClient,
    GeminiSceneLLMClient,
    PromptPolicy,
    build_scene_llm_client,
    build_scene_prompt,
    generate_scene_narrative_with_retries,
)
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

__all__ = [
    "FallbackSceneLLMClient",
    "GeminiSceneLLMClient",
    "LANGGRAPH_AVAILABLE",
    "SceneLLMClient",
    "SceneNarrative",
    "ScenePacket",
    "build_scene_llm_client",
    "build_scene_prompt",
    "run_scene_understanding_pipeline",
    "serialize_scene_packet_toon",
    "sort_scene_narratives",
]


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


def sort_scene_narratives(scene_narratives: list[SceneNarrative]) -> list[SceneNarrative]:
    """Sort scene narratives by chronological scene start time."""
    return sorted(scene_narratives, key=lambda item: item.start_sec)


def _prompt_policy(settings: "Settings") -> PromptPolicy:
    version = getattr(settings, "scene_ai_prompt_version", "v1")
    return PromptPolicy(version=str(version or "v1"))


def _generate_narratives(
    *,
    job_id: str,
    scene_packets: list[ScenePacket],
    llm_client: SceneLLMClient,
    settings: "Settings",
    media_store: "MediaStore",
) -> list[SceneNarrative]:
    narratives: list[SceneNarrative] = []
    prompt_policy = _prompt_policy(settings)
    for packet in scene_packets:
        parsed = generate_scene_narrative_with_retries(
            scene_packet=packet,
            llm_client=llm_client,
            retry_count=settings.scene_llm_retry_count,
            prompt_policy=prompt_policy,
        )

        narrative_key = build_scene_key(job_id, "narrative", packet.scene_id)
        narrative_payload = {
            "scene_id": packet.scene_id,
            "start_sec": packet.start_sec,
            "end_sec": packet.end_sec,
            "narrative_paragraph": parsed.narrative_paragraph,
            "key_moments": parsed.key_moments,
            "packet": packet.packet_key,
        }
        scene_corpus_payload = {
            "scene_id": packet.scene_id,
            "entities": packet.corpus_entities,
            "events": packet.corpus_events,
            "relations": packet.corpus_relations,
            "retrieval_chunks": packet.retrieval_chunks,
            "artifacts": {
                "graph_bundle": packet.graph_bundle_key,
                "retrieval_bundle": packet.retrieval_bundle_key,
            },
        }
        media_store.upload_scene_artifact(
            job_id=job_id,
            artifact_kind="narrative",
            scene_id=packet.scene_id,
            payload=json.dumps(narrative_payload, separators=(",", ":")).encode("utf-8"),
        )
        if hasattr(media_store, "upload_corpus_artifact"):
            media_store.upload_corpus_artifact(
                job_id=job_id,
                artifact_kind="graph",
                payload=json.dumps(scene_corpus_payload, separators=(",", ":")).encode("utf-8"),
                filename=f"scene_{packet.scene_id}.json",
            )
            media_store.upload_corpus_artifact(
                job_id=job_id,
                artifact_kind="retrieval",
                payload=json.dumps(scene_corpus_payload, separators=(",", ":")).encode("utf-8"),
                filename=f"scene_{packet.scene_id}.json",
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
                corpus=scene_corpus_payload | {"narrative_paragraph": parsed.narrative_paragraph},
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
        packets = build_scene_packets(
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
                "corpus": item.corpus,
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

    scene_packets = build_scene_packets(
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
                "corpus": item.corpus,
                "trace": item.trace,
            }
            for item in sort_scene_narratives(scene_narratives)
        ],
        "video_synopsis": video_synopsis,
    }
