"""Tests for SceneGraphDelta validation, repair, and normalization."""

import pytest

from app.scene_graph_delta import (
    EntityDelta,
    EventDelta,
    Evidence,
    Participant,
    RelationDelta,
    SceneGraphDelta,
    normalize_delta,
    validate_delta,
)


def _make_evidence(frames: list[int] | None = None) -> Evidence:
    return Evidence(supporting_frames=frames or [0, 5])


def _make_delta(
    *,
    entities: list[EntityDelta] | None = None,
    relations: list[RelationDelta] | None = None,
    events: list[EventDelta] | None = None,
) -> SceneGraphDelta:
    return SceneGraphDelta(
        scene_id="scene_0",
        entities=entities or [],
        relations=relations or [],
        events=events or [],
    )


ALLOWED_PREDICATES = ["near", "holding", "looking_at", "speaking_to", "wearing"]
SELECTED_FRAMES = [0, 3, 5, 10, 20]
TRACKS_KEYS = {"video_person_7", "object_track_phone_12", "video_person_2"}


class TestValidateDelta:
    def test_valid_delta_passes(self):
        delta = _make_delta(
            entities=[
                EntityDelta(
                    entity_id="video_person_7",
                    type="Person",
                    confidence=0.9,
                    evidence=_make_evidence([0, 5]),
                ),
            ],
            relations=[
                RelationDelta(
                    subject_id="video_person_7",
                    predicate="holding",
                    object_id="object_track_phone_12",
                    time_span_s=(1.0, 2.0),
                    confidence=0.8,
                    evidence=_make_evidence([3, 5]),
                ),
            ],
        )
        errors = validate_delta(
            delta,
            allowed_predicates=ALLOWED_PREDICATES,
            selected_frame_ids=SELECTED_FRAMES,
            tracks_index_keys=TRACKS_KEYS,
        )
        assert errors == []

    def test_invalid_predicate_detected(self):
        delta = _make_delta(
            relations=[
                RelationDelta(
                    subject_id="video_person_7",
                    predicate="punching",
                    object_id="object_track_phone_12",
                    time_span_s=(1.0, 2.0),
                    confidence=0.8,
                    evidence=_make_evidence([0]),
                ),
            ],
        )
        errors = validate_delta(
            delta,
            allowed_predicates=ALLOWED_PREDICATES,
            selected_frame_ids=SELECTED_FRAMES,
            tracks_index_keys=TRACKS_KEYS,
        )
        assert any("punching" in e and "allowed_predicates" in e for e in errors)

    def test_evidence_non_selected_frame_detected(self):
        delta = _make_delta(
            entities=[
                EntityDelta(
                    entity_id="video_person_7",
                    type="Person",
                    confidence=0.9,
                    evidence=_make_evidence([15]),  # frame 15 not in selected
                ),
            ],
        )
        errors = validate_delta(
            delta,
            allowed_predicates=ALLOWED_PREDICATES,
            selected_frame_ids=SELECTED_FRAMES,
            tracks_index_keys=TRACKS_KEYS,
        )
        assert any("frame_id 15" in e for e in errors)

    def test_unknown_entity_id_detected(self):
        delta = _make_delta(
            entities=[
                EntityDelta(
                    entity_id="video_person_99",
                    type="Person",
                    confidence=0.9,
                    evidence=_make_evidence([0]),
                ),
            ],
        )
        errors = validate_delta(
            delta,
            allowed_predicates=ALLOWED_PREDICATES,
            selected_frame_ids=SELECTED_FRAMES,
            tracks_index_keys=TRACKS_KEYS,
        )
        assert any("video_person_99" in e for e in errors)

    def test_unknown_relation_subject_detected(self):
        delta = _make_delta(
            relations=[
                RelationDelta(
                    subject_id="unknown_entity",
                    predicate="near",
                    object_id="video_person_7",
                    time_span_s=(0.0, 1.0),
                    confidence=0.5,
                    evidence=_make_evidence([0]),
                ),
            ],
        )
        errors = validate_delta(
            delta,
            allowed_predicates=ALLOWED_PREDICATES,
            selected_frame_ids=SELECTED_FRAMES,
            tracks_index_keys=TRACKS_KEYS,
        )
        assert any("unknown_entity" in e for e in errors)

    def test_event_participant_unknown_id(self):
        delta = _make_delta(
            events=[
                EventDelta(
                    event_id="ev_1",
                    event_type="handoff",
                    participants=[
                        Participant(entity_id="nonexistent_person", role="giver"),
                    ],
                    time_span_s=(0.0, 1.0),
                    summary="Test event",
                    confidence=0.7,
                    evidence=_make_evidence([0]),
                ),
            ],
        )
        errors = validate_delta(
            delta,
            allowed_predicates=ALLOWED_PREDICATES,
            selected_frame_ids=SELECTED_FRAMES,
            tracks_index_keys=TRACKS_KEYS,
        )
        assert any("nonexistent_person" in e for e in errors)

    def test_multiple_errors_detected(self):
        delta = _make_delta(
            entities=[
                EntityDelta(
                    entity_id="bad_id",
                    type="Person",
                    confidence=0.9,
                    evidence=_make_evidence([99]),
                ),
            ],
            relations=[
                RelationDelta(
                    subject_id="video_person_7",
                    predicate="flying",
                    object_id="bad_object",
                    time_span_s=(0.0, 1.0),
                    confidence=0.5,
                    evidence=_make_evidence([0]),
                ),
            ],
        )
        errors = validate_delta(
            delta,
            allowed_predicates=ALLOWED_PREDICATES,
            selected_frame_ids=SELECTED_FRAMES,
            tracks_index_keys=TRACKS_KEYS,
        )
        # Should have at least: unknown entity, bad evidence frame, bad predicate, bad object
        assert len(errors) >= 3

    def test_empty_delta_passes(self):
        delta = _make_delta()
        errors = validate_delta(
            delta,
            allowed_predicates=ALLOWED_PREDICATES,
            selected_frame_ids=SELECTED_FRAMES,
            tracks_index_keys=TRACKS_KEYS,
        )
        assert errors == []


class TestNormalizeDelta:
    def test_color_normalization(self):
        delta = _make_delta(
            entities=[
                EntityDelta(
                    entity_id="video_person_7",
                    type="Person",
                    attributes={"clothing": ["dark blue jacket"]},
                    confidence=0.9,
                    evidence=_make_evidence(),
                ),
            ],
        )
        normalized = normalize_delta(delta)
        clothing = normalized.entities[0].attributes.get("clothing", [])
        assert isinstance(clothing, list)
        assert len(clothing) > 0
        # "dark blue" should be normalized to "navy"
        assert "navy" in str(clothing[0]).lower()

    def test_ids_not_mutated(self):
        delta = _make_delta(
            entities=[
                EntityDelta(
                    entity_id="video_person_7",
                    type="Person",
                    attributes={"clothing": ["dark blue jacket"]},
                    confidence=0.9,
                    evidence=_make_evidence([0, 5]),
                ),
            ],
            relations=[
                RelationDelta(
                    subject_id="video_person_7",
                    predicate="holding",
                    object_id="object_track_phone_12",
                    time_span_s=(1.0, 2.0),
                    confidence=0.8,
                    evidence=_make_evidence([3]),
                ),
            ],
        )
        normalized = normalize_delta(delta)
        # IDs must remain unchanged
        assert normalized.entities[0].entity_id == "video_person_7"
        assert normalized.relations[0].subject_id == "video_person_7"
        assert normalized.relations[0].predicate == "holding"
        # Evidence frames must remain unchanged
        assert normalized.entities[0].evidence.supporting_frames == [0, 5]

    def test_predicates_not_mutated(self):
        delta = _make_delta(
            relations=[
                RelationDelta(
                    subject_id="a",
                    predicate="near",
                    object_id="b",
                    time_span_s=(0.0, 1.0),
                    confidence=0.5,
                    evidence=_make_evidence(),
                ),
            ],
        )
        normalized = normalize_delta(delta)
        assert normalized.relations[0].predicate == "near"

    def test_normalization_preserves_scene_id(self):
        delta = _make_delta()
        normalized = normalize_delta(delta)
        assert normalized.scene_id == delta.scene_id

    def test_no_crash_without_spacy(self):
        """Normalization should work gracefully even if spaCy model is missing."""
        delta = _make_delta(
            entities=[
                EntityDelta(
                    entity_id="x",
                    type="Object",
                    attributes={"color": "dark blue"},
                    confidence=0.5,
                    evidence=_make_evidence(),
                ),
            ],
        )
        # This should not raise, even if en_core_web_sm isn't installed
        normalized = normalize_delta(delta)
        assert normalized.entities[0].entity_id == "x"
