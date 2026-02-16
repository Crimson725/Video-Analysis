"""Tests for llm_packet: LLM scene packet builder."""

from app.llm_packet import (
    DEFAULT_ALLOWED_PREDICATES,
    LLMPacketConstraints,
    LLMScenePacket,
    build_llm_packet,
)
from app.scene_bundle import (
    DerivedFrameFeatures,
    FrameData,
    SceneBundle,
    TrackMeta,
)


def _make_bundle() -> SceneBundle:
    frames = [
        FrameData(
            frame_id=0,
            timestamp_sec=0.0,
            original_image_key="jobs/test/frames/original/frame_0.jpg",
            overlay_image_key="jobs/test/scene/overlays/scene_0_frame_0.png",
            original_url="https://r2.example.com/frame_0.jpg",
            overlay_url="https://r2.example.com/overlay_0.png",
            tracks=[
                {
                    "track_id": "track_1",
                    "label": "person",
                    "confidence": 0.95,
                    "box": [100, 100, 200, 300],
                    "object_track_id": "obj_track_1",
                },
            ],
            faces=[
                {
                    "face_id": 0,
                    "identity_id": "video_person_7",
                    "confidence": 0.9,
                    "coordinates": [110, 100, 190, 180],
                    "video_person_id": "video_person_7",
                },
            ],
        ),
        FrameData(
            frame_id=5,
            timestamp_sec=2.5,
            original_image_key="jobs/test/frames/original/frame_5.jpg",
            original_url="https://r2.example.com/frame_5.jpg",
            tracks=[],
            faces=[],
        ),
    ]
    tracks_index = {
        "obj_track_1": TrackMeta(
            track_id="obj_track_1",
            label="person",
            entity_type="person",
            confidence_mean=0.95,
            first_frame_id=0,
            last_frame_id=5,
            frame_ids=[0, 5],
        ),
        "video_person_7": TrackMeta(
            track_id="video_person_7",
            label="person",
            entity_type="person",
            confidence_mean=0.9,
            first_frame_id=0,
            last_frame_id=0,
            frame_ids=[0],
            video_person_id="video_person_7",
        ),
    }
    derived = [
        DerivedFrameFeatures(
            frame_id=0,
            centroids={"obj_track_1": (150.0, 200.0)},
            bbox_area_pct={"obj_track_1": 0.5},
        ),
    ]
    return SceneBundle(
        scene_id="0",
        job_id="test_job",
        source_key="input/source.mp4",
        scene_time_span=(0.0, 5.0),
        frames=frames,
        tracks_index=tracks_index,
        derived_features=derived,
        selected_frame_ids=[0, 5],
    )


class TestLLMPacketConstraints:
    def test_defaults(self):
        c = LLMPacketConstraints()
        assert c.evidence_required is True
        assert c.do_not_invent_coordinates is True
        assert c.do_not_override_cv_ids is True
        assert len(c.allowed_predicates) > 0

    def test_custom_predicates(self):
        c = LLMPacketConstraints(allowed_predicates=["near", "holding"])
        assert c.allowed_predicates == ["near", "holding"]


class TestBuildLLMPacket:
    def test_packet_structure(self):
        bundle = _make_bundle()
        packet = build_llm_packet(bundle)

        assert isinstance(packet, LLMScenePacket)
        assert packet.job_id == "test_job"
        assert packet.scene_id == "0"
        assert packet.scene_time_span_s == (0.0, 5.0)
        assert len(packet.frames) == 2
        assert packet.constraints.evidence_required is True

    def test_all_frames_included(self):
        bundle = _make_bundle()
        packet = build_llm_packet(bundle)
        frame_ids = [f["frame_id"] for f in packet.frames]
        assert 0 in frame_ids
        assert 5 in frame_ids

    def test_constraints_present(self):
        bundle = _make_bundle()
        packet = build_llm_packet(bundle)
        assert packet.constraints.evidence_required is True
        assert packet.constraints.do_not_invent_coordinates is True
        assert packet.constraints.do_not_override_cv_ids is True
        assert len(packet.constraints.allowed_predicates) > 0

    def test_custom_allowed_predicates(self):
        bundle = _make_bundle()
        packet = build_llm_packet(bundle, allowed_predicates=["near", "holding"])
        assert packet.constraints.allowed_predicates == ["near", "holding"]

    def test_tracks_index_serialized(self):
        bundle = _make_bundle()
        packet = build_llm_packet(bundle)
        assert "obj_track_1" in packet.tracks_index
        assert packet.tracks_index["obj_track_1"]["label"] == "person"

    def test_derived_features_serialized(self):
        bundle = _make_bundle()
        packet = build_llm_packet(bundle)
        assert len(packet.derived_features) >= 1
        assert packet.derived_features[0]["frame_id"] == 0

    def test_packet_serialization_roundtrip(self):
        bundle = _make_bundle()
        packet = build_llm_packet(bundle)
        json_str = packet.model_dump_json()
        restored = LLMScenePacket.model_validate_json(json_str)
        assert restored.job_id == packet.job_id
        assert len(restored.frames) == len(packet.frames)
