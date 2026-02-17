"""LangGraph KG enrichment workflow: SceneBundle → LLM extraction → Neo4j upsert."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict

from app.llm_packet import DEFAULT_ALLOWED_PREDICATES, build_llm_packet
from app.scene_bundle import build_scene_bundle
from app.scene_graph_delta import (
    SceneGraphDelta,
    normalize_delta,
    repair_delta,
    validate_delta,
)

if TYPE_CHECKING:
    from app.config import Settings
    from app.neo4j_writer import Neo4jWriter
    from app.storage import MediaStore

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    END = START = StateGraph = None  # type: ignore[assignment]


class KGWorkflowState(TypedDict):
    """LangGraph workflow state for the KG enrichment pipeline."""

    job_id: str
    scene_id: int
    source_key: str
    start_sec: float
    end_sec: float
    scene_frames: list[dict[str, Any]]
    scene_bundle: Any  # SceneBundle or None
    llm_packet: dict[str, Any] | None
    llm_raw_output: str | None
    scene_graph_delta: Any  # SceneGraphDelta or None
    validation_errors: list[str]
    normalized_delta: Any  # SceneGraphDelta or None
    neo4j_write_stats: dict[str, Any]
    retry_count: int


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


def extract_scene_graph_delta(
    packet_json: str,
    image_urls: list[str],
    *,
    google_api_key: str,
    model_id: str,
    allowed_predicates: list[str],
) -> tuple[SceneGraphDelta | None, str]:
    """Call the multimodal LLM to extract a SceneGraphDelta.

    Returns (parsed_delta, raw_output_text).
    """
    from pydantic import ValidationError

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        schema = SceneGraphDelta.model_json_schema()
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=google_api_key,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=schema,
        )

        predicates_str = ", ".join(allowed_predicates)
        system_prompt = (
            "You are a scene graph extraction assistant. Analyze the provided scene data "
            "and extract a structured SceneGraphDelta.\n\n"
            "RULES:\n"
            "1. Every entity, relation, and event MUST include evidence with supporting_frames "
            "that reference actual frame_ids from the provided data.\n"
            f"2. Only use these predicates for relations: {predicates_str}\n"
            "3. Do NOT invent coordinates or spatial data not present in the derived features.\n"
            "4. Do NOT override or rename CV-assigned IDs (object_track_id, video_person_id).\n"
            "5. Use entity_id values that match track IDs from the tracks_index.\n"
            "6. If uncertain about an assertion, add it to open_questions instead.\n\n"
            f"Scene packet:\n{packet_json}"
        )

        # Build multimodal content
        content: list[dict[str, Any]] = [{"type": "text", "text": system_prompt}]
        for url in image_urls:
            if url:
                content.append({"type": "image", "url": url})

        messages = [{"role": "user", "content": content}]

        try:
            response = llm.invoke(messages)
        except Exception:
            # Fallback to text-only
            logger.warning("Multimodal invoke failed; falling back to text-only")
            response = llm.invoke(system_prompt)

        raw_text = ""
        if hasattr(response, "content"):
            raw_content = response.content
            if isinstance(raw_content, str):
                raw_text = raw_content
            elif isinstance(raw_content, list):
                parts = []
                for block in raw_content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
                raw_text = "\n".join(parts)
        else:
            raw_text = str(response)

        # Parse JSON
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        parsed = json.loads(text)
        delta = SceneGraphDelta.model_validate(parsed)
        return delta, raw_text

    except (ValidationError, json.JSONDecodeError) as exc:
        logger.warning("LLM extraction parse failed: %s", exc)
        return None, raw_text if "raw_text" in dir() else str(exc)
    except Exception as exc:
        logger.error("LLM extraction failed: %s", exc)
        return None, str(exc)


def _build_repair_prompt(
    validation_errors: list[str],
    original_delta: SceneGraphDelta,
    packet_json: str,
) -> str:
    error_summary = "\n".join(f"- {e}" for e in validation_errors)
    return (
        "The following SceneGraphDelta had validation errors. "
        "Fix the errors and return a corrected JSON that conforms to the SceneGraphDelta schema.\n\n"
        f"Validation errors:\n{error_summary}\n\n"
        f"Original delta:\n{original_delta.model_dump_json(indent=2)}\n\n"
        f"Scene context:\n{packet_json}\n\n"
        "Return ONLY the corrected SceneGraphDelta JSON."
    )


def repair_scene_graph_delta(
    delta: SceneGraphDelta,
    validation_errors: list[str],
    packet_json: str,
    *,
    google_api_key: str,
    model_id: str,
) -> tuple[SceneGraphDelta | None, str]:
    """Attempt to repair a failed delta by sending errors back to the LLM."""
    from pydantic import ValidationError

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        schema = SceneGraphDelta.model_json_schema()
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=google_api_key,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=schema,
        )

        prompt = _build_repair_prompt(validation_errors, delta, packet_json)
        response = llm.invoke(prompt)
        raw_text = ""
        if hasattr(response, "content"):
            raw_text = str(response.content)
        else:
            raw_text = str(response)

        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        parsed = json.loads(text)
        repaired = SceneGraphDelta.model_validate(parsed)
        return repaired, raw_text

    except (ValidationError, json.JSONDecodeError) as exc:
        logger.warning("Repair parse failed: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.error("Repair LLM call failed: %s", exc)
        return None, str(exc)


# ---------------------------------------------------------------------------
# Workflow node functions
# ---------------------------------------------------------------------------


def _build_scene_bundle_node(
    state: KGWorkflowState,
    settings: "Settings",
    media_store: "MediaStore",
) -> dict[str, Any]:
    """BuildSceneBundle node."""
    bundle = build_scene_bundle(
        job_id=state["job_id"],
        scene_id=state["scene_id"],
        source_key=state["source_key"],
        start_sec=state["start_sec"],
        end_sec=state["end_sec"],
        scene_frames=state["scene_frames"],
        media_store=media_store,
        max_keyframes=settings.kg_max_keyframes,
        motion_threshold=settings.kg_motion_threshold,
        interaction_distance_threshold=settings.kg_interaction_distance_threshold,
        near_threshold=settings.kg_near_threshold,
    )
    return {"scene_bundle": bundle}


def _build_llm_packet_node(
    state: KGWorkflowState,
    settings: "Settings",
) -> dict[str, Any]:
    """BuildLLMPacket node."""
    bundle = state["scene_bundle"]
    if bundle is None:
        return {"llm_packet": None}
    predicates = None
    if settings.kg_allowed_predicates:
        predicates = [
            p.strip() for p in settings.kg_allowed_predicates.split(",") if p.strip()
        ]
    packet = build_llm_packet(bundle, allowed_predicates=predicates)
    return {"llm_packet": packet.model_dump()}


def _llm_extract_delta_node(
    state: KGWorkflowState,
    settings: "Settings",
) -> dict[str, Any]:
    """LLMExtractDelta node."""
    packet = state.get("llm_packet")
    bundle = state.get("scene_bundle")
    if packet is None or bundle is None:
        return {
            "scene_graph_delta": None,
            "llm_raw_output": "No packet available",
            "validation_errors": ["No LLM packet available"],
        }

    packet_json = json.dumps(packet, default=str)

    # Collect image URLs from bundle
    image_urls: list[str] = []
    for frame in bundle.frames:
        if frame.original_url:
            image_urls.append(frame.original_url)
        if frame.overlay_url:
            image_urls.append(frame.overlay_url)

    allowed = packet.get("constraints", {}).get(
        "allowed_predicates", DEFAULT_ALLOWED_PREDICATES
    )

    delta, raw_output = extract_scene_graph_delta(
        packet_json=packet_json,
        image_urls=image_urls,
        google_api_key=settings.google_api_key,
        model_id=settings.scene_model_id,
        allowed_predicates=allowed,
    )

    errors: list[str] = []
    if delta is None:
        errors = [f"LLM extraction failed: {raw_output[:200]}"]

    return {
        "scene_graph_delta": delta,
        "llm_raw_output": raw_output,
        "validation_errors": errors,
    }


def _validate_delta_node(
    state: KGWorkflowState,
    settings: "Settings",
) -> dict[str, Any]:
    """ValidateDelta node."""
    delta = state.get("scene_graph_delta")
    bundle = state.get("scene_bundle")

    if delta is None:
        return {
            "validation_errors": state.get(
                "validation_errors", ["No delta to validate"]
            )
        }

    if bundle is None:
        return {"validation_errors": ["No scene bundle available for validation"]}

    allowed = DEFAULT_ALLOWED_PREDICATES
    if settings.kg_allowed_predicates:
        allowed = [
            p.strip() for p in settings.kg_allowed_predicates.split(",") if p.strip()
        ]

    errors = validate_delta(
        delta,
        allowed_predicates=allowed,
        selected_frame_ids=bundle.selected_frame_ids,
        tracks_index_keys=set(bundle.tracks_index.keys()),
    )
    return {"validation_errors": errors}


def _repair_delta_node(
    state: KGWorkflowState,
    settings: "Settings",
) -> dict[str, Any]:
    """RepairDelta node."""
    delta = state.get("scene_graph_delta")
    errors = state.get("validation_errors", [])
    packet = state.get("llm_packet")

    if delta is None or not errors:
        return {"retry_count": state.get("retry_count", 0) + 1}

    packet_json = json.dumps(packet, default=str) if packet else "{}"

    repaired, raw = repair_scene_graph_delta(
        delta=delta,
        validation_errors=errors,
        packet_json=packet_json,
        google_api_key=settings.google_api_key,
        model_id=settings.scene_model_id,
    )

    return {
        "scene_graph_delta": repaired if repaired else delta,
        "llm_raw_output": raw,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _normalize_delta_node(state: KGWorkflowState) -> dict[str, Any]:
    """NormalizeDelta node."""
    delta = state.get("scene_graph_delta")
    if delta is None:
        return {"normalized_delta": None}
    normalized = normalize_delta(delta)
    return {"normalized_delta": normalized}


def qualify_delta_ids(
    delta: SceneGraphDelta,
    job_id: str,
    scene_index: int,
    source_key: str,
) -> dict[str, Any]:
    """Compute fully-qualified UIDs for all delta contents.

    Returns a kwargs dict suitable for ``Neo4jWriter.upsert_scene``.

    UID formulas (colon-separated):
    - scene_uid  = ``{job_id}:{scene_index}``
    - frame_uid  = ``{scene_uid}:{frame_id}``
    - entity_uid = ``{job_id}:{entity_local_id}``
    - event_uid  = ``{scene_uid}:{event_id}``
    """
    scene_uid = f"{job_id}:{scene_index}"

    # Entities
    entities: list[dict[str, Any]] = []
    for e in delta.entities:
        entities.append(
            {
                "entity_uid": f"{job_id}:{e.entity_id}",
                "entity_local_id": e.entity_id,
                "type": e.type,
                "attributes": dict(e.attributes),
                "confidence": e.confidence,
            }
        )

    # Relations — carry subject_uid / object_uid
    relations: list[dict[str, Any]] = []
    for r in delta.relations:
        relations.append(
            {
                "subject_uid": f"{job_id}:{r.subject_id}",
                "object_uid": f"{job_id}:{r.object_id}",
                "predicate": r.predicate,
                "time_span_s": list(r.time_span_s),
                "confidence": r.confidence,
                "evidence": r.evidence.model_dump(),
            }
        )

    # Events — carry event_uid and participant entity_uids
    events: list[dict[str, Any]] = []
    for ev in delta.events:
        events.append(
            {
                "event_uid": f"{scene_uid}:{ev.event_id}",
                "event_id": ev.event_id,
                "event_type": ev.event_type,
                "participants": [
                    {
                        "entity_uid": f"{job_id}:{p.entity_id}",
                        "entity_id": p.entity_id,
                        "role": p.role,
                    }
                    for p in ev.participants
                ],
                "time_span_s": list(ev.time_span_s),
                "summary": ev.summary,
                "confidence": ev.confidence,
            }
        )

    return {
        "video_id": job_id,
        "scene_uid": scene_uid,
        "scene_index": scene_index,
        "job_id": job_id,
        "source_key": source_key,
        "entities": entities,
        "relations": relations,
        "events": events,
    }


def _merge_upsert_neo4j_node(
    state: KGWorkflowState,
    neo4j_writer: "Neo4jWriter | None",
) -> dict[str, Any]:
    """MergeAndUpsertNeo4j node."""
    delta = state.get("normalized_delta") or state.get("scene_graph_delta")
    bundle = state.get("scene_bundle")

    if delta is None or bundle is None:
        return {"neo4j_write_stats": {"error": "No data to upsert"}}

    if neo4j_writer is None:
        return {"neo4j_write_stats": {"error": "No Neo4j writer available"}}

    scene_index = int(bundle.scene_id)

    # Compute fully-qualified UIDs via the qualifier
    qualified = qualify_delta_ids(
        delta,
        job_id=bundle.job_id,
        scene_index=scene_index,
        source_key=bundle.source_key,
    )

    # Frame timestamps and UIDs
    frame_timestamps = {f.frame_id: f.timestamp_sec for f in bundle.frames}
    scene_uid = qualified["scene_uid"]
    frame_uids = {fid: f"{scene_uid}:{fid}" for fid in bundle.selected_frame_ids}

    metrics = neo4j_writer.upsert_scene(
        video_id=qualified["video_id"],
        scene_uid=qualified["scene_uid"],
        scene_index=qualified["scene_index"],
        job_id=qualified["job_id"],
        source_key=qualified["source_key"],
        t0=bundle.scene_time_span[0],
        t1=bundle.scene_time_span[1],
        selected_frame_ids=bundle.selected_frame_ids,
        frame_timestamps=frame_timestamps,
        frame_uids=frame_uids,
        entities=qualified["entities"],
        relations=qualified["relations"],
        events=qualified["events"],
    )

    return {
        "neo4j_write_stats": {
            "nodes_created": metrics.nodes_created,
            "nodes_updated": metrics.nodes_updated,
            "relationships_created": metrics.relationships_created,
            "relationships_updated": metrics.relationships_updated,
            "errors": metrics.errors,
        }
    }


# ---------------------------------------------------------------------------
# Validation routing
# ---------------------------------------------------------------------------


def _validation_router(state: KGWorkflowState, settings: "Settings") -> str:
    """Route after validation: normalize (pass), repair (fail+retries), or end."""
    errors = state.get("validation_errors", [])
    if not errors:
        return "normalize"
    max_retries = settings.kg_max_repair_retries
    if state.get("retry_count", 0) < max_retries:
        return "repair"
    logger.warning(
        "KG validation failed for scene %s after %d retries: %s",
        state.get("scene_id"),
        state.get("retry_count", 0),
        errors,
    )
    return "end"


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------


def build_kg_workflow(
    settings: "Settings",
    media_store: "MediaStore",
    neo4j_writer: "Neo4jWriter | None" = None,
) -> Any:
    """Build and compile the KG enrichment LangGraph workflow."""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph is not available")

    def build_bundle(state: KGWorkflowState) -> dict[str, Any]:
        return _build_scene_bundle_node(state, settings, media_store)

    def build_packet(state: KGWorkflowState) -> dict[str, Any]:
        return _build_llm_packet_node(state, settings)

    def extract_delta(state: KGWorkflowState) -> dict[str, Any]:
        return _llm_extract_delta_node(state, settings)

    def validate(state: KGWorkflowState) -> dict[str, Any]:
        return _validate_delta_node(state, settings)

    def repair(state: KGWorkflowState) -> dict[str, Any]:
        return _repair_delta_node(state, settings)

    def normalize(state: KGWorkflowState) -> dict[str, Any]:
        return _normalize_delta_node(state)

    def upsert(state: KGWorkflowState) -> dict[str, Any]:
        return _merge_upsert_neo4j_node(state, neo4j_writer)

    def router(state: KGWorkflowState) -> str:
        return _validation_router(state, settings)

    graph_builder = StateGraph(KGWorkflowState)

    graph_builder.add_node("build_bundle", build_bundle)
    graph_builder.add_node("build_packet", build_packet)
    graph_builder.add_node("extract_delta", extract_delta)
    graph_builder.add_node("validate", validate)
    graph_builder.add_node("repair", repair)
    graph_builder.add_node("normalize", normalize)
    graph_builder.add_node("upsert", upsert)

    graph_builder.add_edge(START, "build_bundle")
    graph_builder.add_edge("build_bundle", "build_packet")
    graph_builder.add_edge("build_packet", "extract_delta")
    graph_builder.add_edge("extract_delta", "validate")
    graph_builder.add_conditional_edges(
        "validate",
        router,
        {
            "normalize": "normalize",
            "repair": "repair",
            "end": END,
        },
    )
    graph_builder.add_edge("repair", "validate")
    graph_builder.add_edge("normalize", "upsert")
    graph_builder.add_edge("upsert", END)

    return graph_builder.compile()


def run_kg_workflow(
    *,
    job_id: str,
    scene_id: int,
    source_key: str,
    start_sec: float,
    end_sec: float,
    scene_frames: list[dict[str, Any]],
    settings: "Settings",
    media_store: "MediaStore",
    neo4j_writer: "Neo4jWriter | None" = None,
) -> dict[str, Any]:
    """Execute the KG enrichment workflow for one scene."""
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available; skipping KG workflow")
        return {
            "neo4j_write_stats": {},
            "validation_errors": ["LangGraph not available"],
        }

    graph = build_kg_workflow(settings, media_store, neo4j_writer)

    initial_state: KGWorkflowState = {
        "job_id": job_id,
        "scene_id": scene_id,
        "source_key": source_key,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "scene_frames": scene_frames,
        "scene_bundle": None,
        "llm_packet": None,
        "llm_raw_output": None,
        "scene_graph_delta": None,
        "validation_errors": [],
        "normalized_delta": None,
        "neo4j_write_stats": {},
        "retry_count": 0,
    }

    output = graph.invoke(initial_state)

    return {
        "neo4j_write_stats": output.get("neo4j_write_stats", {}),
        "validation_errors": output.get("validation_errors", []),
        "scene_graph_delta": output.get("scene_graph_delta"),
        "normalized_delta": output.get("normalized_delta"),
        "llm_raw_output": output.get("llm_raw_output"),
        "retry_count": output.get("retry_count", 0),
    }
