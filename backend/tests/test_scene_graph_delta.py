"""Tests for SceneGraphDelta Pydantic model validation."""

import pytest
from pydantic import ValidationError

from app.scene_graph_delta import (
    EntityDelta,
    EventDelta,
    Evidence,
    Participant,
    RelationDelta,
    SceneGraphDelta,
)


class TestEvidence:
    def test_valid_evidence(self):
        e = Evidence(supporting_frames=[0, 5, 10], supporting_track_ids=["track_a"])
        assert e.supporting_frames == [0, 5, 10]
        assert e.supporting_track_ids == ["track_a"]

    def test_empty_supporting_frames_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(supporting_frames=[])

    def test_default_track_ids_empty(self):
        e = Evidence(supporting_frames=[1])
        assert e.supporting_track_ids == []


class TestEntityDelta:
    def test_valid_entity(self):
        entity = EntityDelta(
            entity_id="video_person_7",
            type="Person",
            attributes={"clothing": ["dark jacket"], "emotion": "neutral"},
            confidence=0.85,
            evidence=Evidence(supporting_frames=[0, 5]),
        )
        assert entity.entity_id == "video_person_7"
        assert entity.type == "Person"
        assert entity.attributes["clothing"] == ["dark jacket"]

    def test_missing_evidence_rejected(self):
        with pytest.raises(ValidationError):
            EntityDelta(
                entity_id="video_person_7",
                type="Person",
                confidence=0.85,
                # evidence is missing
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            EntityDelta(
                entity_id="video_person_7",
                type="Person",
                confidence=1.5,
                evidence=Evidence(supporting_frames=[1]),
            )

    def test_negative_confidence_rejected(self):
        with pytest.raises(ValidationError):
            EntityDelta(
                entity_id="x",
                type="Object",
                confidence=-0.1,
                evidence=Evidence(supporting_frames=[1]),
            )


class TestRelationDelta:
    def test_valid_relation(self):
        rel = RelationDelta(
            subject_id="video_person_7",
            predicate="holding",
            object_id="object_track_phone_12",
            time_span_s=(1.0, 2.4),
            confidence=0.9,
            evidence=Evidence(supporting_frames=[3, 5]),
        )
        assert rel.predicate == "holding"
        assert rel.time_span_s == (1.0, 2.4)

    def test_missing_predicate_rejected(self):
        with pytest.raises(ValidationError):
            RelationDelta(
                subject_id="a",
                object_id="b",
                time_span_s=(0.0, 1.0),
                confidence=0.5,
                evidence=Evidence(supporting_frames=[1]),
            )


class TestEventDelta:
    def test_valid_event(self):
        event = EventDelta(
            event_id="event_001",
            event_type="handoff_object",
            participants=[
                Participant(entity_id="video_person_7", role="giver"),
                Participant(entity_id="video_person_2", role="receiver"),
            ],
            time_span_s=(1.8, 2.2),
            summary="Person 7 hands an object to Person 2",
            confidence=0.8,
            evidence=Evidence(supporting_frames=[4, 5]),
        )
        assert event.event_type == "handoff_object"
        assert len(event.participants) == 2


class TestSceneGraphDelta:
    def test_valid_full_delta(self):
        delta = SceneGraphDelta(
            scene_id="scene_0003",
            entities=[
                EntityDelta(
                    entity_id="video_person_7",
                    type="Person",
                    attributes={"clothing": ["dark jacket"]},
                    confidence=0.9,
                    evidence=Evidence(supporting_frames=[0, 5]),
                ),
                EntityDelta(
                    entity_id="object_track_phone_12",
                    type="Object",
                    attributes={"color": "black"},
                    confidence=0.85,
                    evidence=Evidence(supporting_frames=[3]),
                ),
            ],
            relations=[
                RelationDelta(
                    subject_id="video_person_7",
                    predicate="holding",
                    object_id="object_track_phone_12",
                    time_span_s=(1.0, 2.4),
                    confidence=0.9,
                    evidence=Evidence(supporting_frames=[3, 5]),
                ),
            ],
            events=[
                EventDelta(
                    event_id="event_001",
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
            scene_summary="A person holds a phone and hands it off.",
            scene_tags=["interaction", "handoff"],
            open_questions=[{"question": "Is person 7 the owner of the phone?"}],
        )
        assert delta.scene_id == "scene_0003"
        assert len(delta.entities) == 2
        assert len(delta.relations) == 1
        assert len(delta.events) == 1
        assert delta.scene_summary is not None
        assert len(delta.scene_tags) == 2

    def test_minimal_delta(self):
        delta = SceneGraphDelta(scene_id="scene_0")
        assert delta.entities == []
        assert delta.relations == []
        assert delta.events == []
        assert delta.scene_summary is None
        assert delta.scene_tags == []

    def test_missing_scene_id_rejected(self):
        with pytest.raises(ValidationError):
            SceneGraphDelta()

    def test_invalid_entity_in_delta_rejected(self):
        with pytest.raises(ValidationError):
            SceneGraphDelta(
                scene_id="scene_0",
                entities=[
                    {"entity_id": "x", "type": "Person", "confidence": 0.9}
                    # missing evidence
                ],
            )

    def test_serialization_roundtrip(self):
        delta = SceneGraphDelta(
            scene_id="scene_1",
            entities=[
                EntityDelta(
                    entity_id="track_1",
                    type="Object",
                    confidence=0.7,
                    evidence=Evidence(supporting_frames=[0]),
                ),
            ],
        )
        json_str = delta.model_dump_json()
        restored = SceneGraphDelta.model_validate_json(json_str)
        assert restored.scene_id == "scene_1"
        assert len(restored.entities) == 1
        assert restored.entities[0].entity_id == "track_1"
