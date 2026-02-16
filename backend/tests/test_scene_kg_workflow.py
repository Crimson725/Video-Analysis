"""Tests for scene_kg_workflow: KG enrichment LangGraph workflow."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.scene_bundle import SceneBundle, build_scene_bundle
from app.llm_packet import build_llm_packet
from app.scene_graph_delta import (
    EntityDelta,
    Evidence,
    RelationDelta,
    SceneGraphDelta,
    validate_delta,
)
from app.scene_kg_workflow import (
    KGWorkflowState,
    _build_llm_packet_node,
    _build_scene_bundle_node,
    _normalize_delta_node,
    _validate_delta_node,
    _validation_router,
    qualify_delta_ids,
)


def _make_settings(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "kg_max_keyframes": 8,
        "kg_motion_threshold": 80.0,
        "kg_interaction_distance_threshold": 150.0,
        "kg_near_threshold": 150.0,
        "kg_max_repair_retries": 1,
        "kg_allowed_predicates": "",
        "kg_pipeline_enabled": True,
        "google_api_key": "test-key",
        "scene_model_id": "gemini-3-flash-preview",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_frame(frame_id: int, timestamp: str = "00:00:00.000", **kwargs: Any) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "files": {
            "original": f"jobs/test/frames/original/frame_{frame_id}.jpg",
            "detection": f"jobs/test/frames/det/frame_{frame_id}.jpg",
            "segmentation": f"jobs/test/frames/seg/frame_{frame_id}.jpg",
            "face": f"jobs/test/frames/face/frame_{frame_id}.jpg",
        },
        "analysis": {
            "object_detection": kwargs.get("detections", [
                {
                    "track_id": "track_1",
                    "label": "person",
                    "confidence": 0.95,
                    "box": [100, 100, 200, 300],
                    "object_track_id": "obj_track_1",
                },
            ]),
            "face_recognition": kwargs.get("faces", []),
            "semantic_segmentation": kwargs.get("segmentation", []),
        },
        "analysis_artifacts": {"json": f"jobs/test/analysis/json/frame_{frame_id}.json"},
    }


def _make_scene_frames(n: int = 5) -> list[dict]:
    return [_make_frame(i, f"00:00:{i:02d}.000") for i in range(n)]


def _make_initial_state(**overrides: Any) -> KGWorkflowState:
    defaults: KGWorkflowState = {
        "job_id": "test_job",
        "scene_id": 0,
        "source_key": "input/source.mp4",
        "start_sec": 0.0,
        "end_sec": 5.0,
        "scene_frames": _make_scene_frames(),
        "scene_bundle": None,
        "llm_packet": None,
        "llm_raw_output": None,
        "scene_graph_delta": None,
        "validation_errors": [],
        "normalized_delta": None,
        "neo4j_write_stats": {},
        "retry_count": 0,
    }
    defaults.update(overrides)  # type: ignore[arg-type]
    return defaults


class TestBuildSceneBundleNode:
    def test_builds_bundle(self):
        state = _make_initial_state()
        settings = _make_settings()
        result = _build_scene_bundle_node(state, settings, media_store=None)
        assert "scene_bundle" in result
        bundle = result["scene_bundle"]
        assert isinstance(bundle, SceneBundle)
        assert bundle.job_id == "test_job"
        assert len(bundle.frames) > 0


class TestBuildLLMPacketNode:
    def test_builds_packet(self):
        state = _make_initial_state()
        settings = _make_settings()
        # First build bundle
        bundle_result = _build_scene_bundle_node(state, settings, media_store=None)
        state["scene_bundle"] = bundle_result["scene_bundle"]
        # Then build packet
        result = _build_llm_packet_node(state, settings)
        assert "llm_packet" in result
        packet = result["llm_packet"]
        assert isinstance(packet, dict)
        assert "frames" in packet
        assert "constraints" in packet
        assert packet["constraints"]["evidence_required"] is True


class TestValidateDeltaNode:
    def test_valid_delta_passes(self):
        state = _make_initial_state()
        settings = _make_settings()
        bundle_result = _build_scene_bundle_node(state, settings, media_store=None)
        bundle = bundle_result["scene_bundle"]
        state["scene_bundle"] = bundle

        # Create a delta that uses valid track IDs and frame IDs
        track_ids = list(bundle.tracks_index.keys())
        frame_ids = bundle.selected_frame_ids

        delta = SceneGraphDelta(
            scene_id="0",
            entities=[
                EntityDelta(
                    entity_id=track_ids[0] if track_ids else "obj_track_1",
                    type="Object",
                    confidence=0.9,
                    evidence=Evidence(
                        supporting_frames=[frame_ids[0]] if frame_ids else [0]
                    ),
                ),
            ],
        )
        state["scene_graph_delta"] = delta
        result = _validate_delta_node(state, settings)
        assert result["validation_errors"] == []

    def test_invalid_delta_detected(self):
        state = _make_initial_state()
        settings = _make_settings()
        bundle_result = _build_scene_bundle_node(state, settings, media_store=None)
        state["scene_bundle"] = bundle_result["scene_bundle"]

        delta = SceneGraphDelta(
            scene_id="0",
            entities=[
                EntityDelta(
                    entity_id="nonexistent_track",
                    type="Person",
                    confidence=0.9,
                    evidence=Evidence(supporting_frames=[999]),
                ),
            ],
        )
        state["scene_graph_delta"] = delta
        result = _validate_delta_node(state, settings)
        assert len(result["validation_errors"]) > 0


class TestValidationRouter:
    def test_routes_to_normalize_on_success(self):
        state = _make_initial_state(validation_errors=[], retry_count=0)
        settings = _make_settings()
        assert _validation_router(state, settings) == "normalize"

    def test_routes_to_repair_on_first_failure(self):
        state = _make_initial_state(
            validation_errors=["some error"],
            retry_count=0,
        )
        settings = _make_settings(kg_max_repair_retries=1)
        assert _validation_router(state, settings) == "repair"

    def test_routes_to_end_when_retries_exhausted(self):
        state = _make_initial_state(
            validation_errors=["persistent error"],
            retry_count=1,
        )
        settings = _make_settings(kg_max_repair_retries=1)
        assert _validation_router(state, settings) == "end"


class TestNormalizeDeltaNode:
    def test_normalizes_delta(self):
        delta = SceneGraphDelta(
            scene_id="0",
            entities=[
                EntityDelta(
                    entity_id="track_1",
                    type="Person",
                    attributes={"clothing": ["dark blue jacket"]},
                    confidence=0.9,
                    evidence=Evidence(supporting_frames=[0]),
                ),
            ],
        )
        state = _make_initial_state(scene_graph_delta=delta)
        result = _normalize_delta_node(state)
        assert "normalized_delta" in result
        normalized = result["normalized_delta"]
        assert normalized is not None
        assert normalized.entities[0].entity_id == "track_1"

    def test_handles_none_delta(self):
        state = _make_initial_state(scene_graph_delta=None)
        result = _normalize_delta_node(state)
        assert result["normalized_delta"] is None


class TestQualifyDeltaIds:
    """Unit tests for qualify_delta_ids UID computation."""

    def _sample_delta(self) -> SceneGraphDelta:
        from app.scene_graph_delta import EventDelta, Participant, RelationDelta

        return SceneGraphDelta(
            scene_id="0",
            entities=[
                EntityDelta(
                    entity_id="video_person_7",
                    type="Person",
                    attributes={"clothing": ["dark jacket"]},
                    confidence=0.9,
                    evidence=Evidence(supporting_frames=[0, 5]),
                ),
                EntityDelta(
                    entity_id="obj_track_phone",
                    type="Object",
                    attributes={"label": "phone"},
                    confidence=0.85,
                    evidence=Evidence(supporting_frames=[3]),
                ),
            ],
            relations=[
                RelationDelta(
                    subject_id="video_person_7",
                    predicate="holding",
                    object_id="obj_track_phone",
                    time_span_s=(1.0, 2.4),
                    confidence=0.9,
                    evidence=Evidence(supporting_frames=[3, 5]),
                ),
            ],
            events=[
                EventDelta(
                    event_id="ev_001",
                    event_type="handoff_object",
                    participants=[
                        Participant(entity_id="video_person_7", role="giver"),
                    ],
                    time_span_s=(1.8, 2.2),
                    summary="Object handoff",
                    confidence=0.8,
                    evidence=Evidence(supporting_frames=[4]),
                ),
            ],
        )

    def test_scene_uid_format(self):
        result = qualify_delta_ids(self._sample_delta(), "job-abc", 0, "input/v.mp4")
        assert result["scene_uid"] == "job-abc:0"

    def test_entity_uid_format(self):
        result = qualify_delta_ids(self._sample_delta(), "job-abc", 0, "input/v.mp4")
        uids = [e["entity_uid"] for e in result["entities"]]
        assert "job-abc:video_person_7" in uids
        assert "job-abc:obj_track_phone" in uids
        # Local IDs preserved
        local_ids = [e["entity_local_id"] for e in result["entities"]]
        assert "video_person_7" in local_ids
        assert "obj_track_phone" in local_ids

    def test_event_uid_format(self):
        result = qualify_delta_ids(self._sample_delta(), "job-abc", 2, "input/v.mp4")
        assert result["events"][0]["event_uid"] == "job-abc:2:ev_001"
        assert result["events"][0]["event_id"] == "ev_001"

    def test_relation_subject_object_uids(self):
        result = qualify_delta_ids(self._sample_delta(), "job-abc", 0, "input/v.mp4")
        rel = result["relations"][0]
        assert rel["subject_uid"] == "job-abc:video_person_7"
        assert rel["object_uid"] == "job-abc:obj_track_phone"

    def test_event_participant_entity_uid(self):
        result = qualify_delta_ids(self._sample_delta(), "job-abc", 0, "input/v.mp4")
        participant = result["events"][0]["participants"][0]
        assert participant["entity_uid"] == "job-abc:video_person_7"

    def test_video_id_equals_job_id(self):
        result = qualify_delta_ids(self._sample_delta(), "my-job", 0, "input/v.mp4")
        assert result["video_id"] == "my-job"
        assert result["job_id"] == "my-job"
