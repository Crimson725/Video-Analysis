"""Tests for app.schemas — Pydantic model validation and serialization."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnalysisArtifacts,
    DetectionItem,
    FaceItem,
    FrameAnalysis,
    FrameFiles,
    FrameResult,
    JobResult,
    SegmentationItem,
    SceneEntity,
    SceneTemporalSpan,
)


class TestSegmentationItem:
    def test_class_alias_serialization(self):
        item = SegmentationItem(
            object_id=1, class_name="person", mask_polygon=[[0, 0], [1, 1]]
        )
        data = item.model_dump(by_alias=True)
        assert "class" in data
        assert data["class"] == "person"
        assert "class_name" not in data

    def test_populate_by_name(self):
        item = SegmentationItem(object_id=1, class_name="car", mask_polygon=[[0, 0]])
        assert item.class_name == "car"

    def test_accepts_rgb_color_triplets(self):
        item = SegmentationItem(
            object_id=1,
            class_name="person",
            mask_polygon=[[0, 0], [1, 1]],
            palette_rgb=[4, 42, 255],
            bbox_rgb=[4, 42, 255],
        )
        assert item.palette_rgb == [4, 42, 255]
        assert item.bbox_rgb == [4, 42, 255]

    def test_rejects_invalid_palette_rgb_length(self):
        with pytest.raises(ValidationError):
            SegmentationItem(
                object_id=1,
                class_name="person",
                mask_polygon=[[0, 0], [1, 1]],
                palette_rgb=[4, 42],
            )


class TestDetectionItem:
    def test_valid_box(self):
        item = DetectionItem(
            track_id="dog_1", label="dog", confidence=0.95, box=[10, 20, 30, 40]
        )
        assert item.box == [10, 20, 30, 40]

    def test_rejects_box_with_3_elements(self):
        with pytest.raises(ValidationError):
            DetectionItem(
                track_id="dog_1", label="dog", confidence=0.95, box=[10, 20, 30]
            )

    def test_rejects_box_with_5_elements(self):
        with pytest.raises(ValidationError):
            DetectionItem(
                track_id="dog_1", label="dog", confidence=0.95, box=[10, 20, 30, 40, 50]
            )

    def test_track_id_is_required(self):
        with pytest.raises(ValidationError):
            DetectionItem(label="dog", confidence=0.95, box=[10, 20, 30, 40])  # type: ignore[call-arg]

    def test_accepts_rgb_color_triplets(self):
        item = DetectionItem(
            track_id="dog_1",
            label="dog",
            confidence=0.95,
            box=[10, 20, 30, 40],
            palette_rgb=[255, 111, 221],
            bbox_rgb=[255, 111, 221],
        )
        assert item.palette_rgb == [255, 111, 221]
        assert item.bbox_rgb == [255, 111, 221]

    def test_rejects_invalid_bbox_rgb_length(self):
        with pytest.raises(ValidationError):
            DetectionItem(
                track_id="dog_1",
                label="dog",
                confidence=0.95,
                box=[10, 20, 30, 40],
                bbox_rgb=[255, 111],
            )

    def test_accepts_fused_person_tracking_fields(self):
        item = DetectionItem(
            track_id="person_4",
            label="person",
            confidence=0.93,
            box=[10, 20, 30, 40],
            object_track_id="object_track_abc",
            object_identity_confidence=0.82,
            object_identity_is_ambiguous=False,
            person_track_id="person_track_123",
            person_identity_id="video_person_1",
            person_identity_source="video_person_id",
            person_identity_confidence=0.91,
            person_identity_is_ambiguous=False,
        )

        assert item.object_track_id == "object_track_abc"
        assert item.object_identity_confidence == pytest.approx(0.82)
        assert item.object_identity_is_ambiguous is False
        assert item.person_track_id == "person_track_123"
        assert item.person_identity_id == "video_person_1"
        assert item.person_identity_source == "video_person_id"
        assert item.person_identity_confidence == pytest.approx(0.91)
        assert item.person_identity_is_ambiguous is False


class TestFaceItem:
    def test_valid_coordinates(self):
        item = FaceItem(
            face_id=1,
            identity_id="face_1",
            confidence=0.99,
            coordinates=[10, 20, 30, 40],
        )
        assert item.identity_id == "face_1"
        assert item.coordinates == [10, 20, 30, 40]

    def test_rejects_coordinates_with_5_elements(self):
        with pytest.raises(ValidationError):
            FaceItem(
                face_id=1,
                identity_id="face_1",
                confidence=0.99,
                coordinates=[10, 20, 30, 40, 50],
            )

    def test_rejects_coordinates_with_3_elements(self):
        with pytest.raises(ValidationError):
            FaceItem(
                face_id=1,
                identity_id="face_1",
                confidence=0.99,
                coordinates=[10, 20, 30],
            )

    def test_identity_id_is_required(self):
        with pytest.raises(ValidationError):
            FaceItem(  # type: ignore[call-arg]
                face_id=1,
                confidence=0.99,
                coordinates=[10, 20, 30, 40],
            )

    def test_accepts_identity_metadata_fields(self):
        item = FaceItem(
            face_id=1,
            identity_id="face_1",
            confidence=0.99,
            coordinates=[10, 20, 30, 40],
            palette_rgb=[11, 219, 235],
            bbox_rgb=[11, 219, 235],
            scene_person_id="scene_0_person_1",
            video_person_id="video_person_1",
            match_confidence=0.92,
            is_identity_ambiguous=False,
            embedding_model_id="edgeface_s_gamma_05",
        )

        assert item.scene_person_id == "scene_0_person_1"
        assert item.video_person_id == "video_person_1"
        assert item.match_confidence == pytest.approx(0.92)
        assert item.is_identity_ambiguous is False
        assert item.embedding_model_id == "edgeface_s_gamma_05"
        assert item.palette_rgb == [11, 219, 235]
        assert item.bbox_rgb == [11, 219, 235]

    def test_rejects_invalid_palette_rgb_length(self):
        with pytest.raises(ValidationError):
            FaceItem(
                face_id=1,
                identity_id="face_1",
                confidence=0.99,
                coordinates=[10, 20, 30, 40],
                palette_rgb=[11, 219],
            )


class TestJobResult:
    def test_valid_construction(self):
        result = JobResult(
            job_id="abc-123",
            frames=[
                FrameResult(
                    frame_id=0,
                    timestamp="00:00:05.000",
                    raw_frame_index=125,
                    files=FrameFiles(
                        original="https://example.com/jobs/abc/frames/original/frame_0.jpg?sig=1",
                        segmentation="https://example.com/jobs/abc/frames/seg/frame_0.jpg?sig=1",
                        detection="https://example.com/jobs/abc/frames/det/frame_0.jpg?sig=1",
                        face="https://example.com/jobs/abc/frames/face/frame_0.jpg?sig=1",
                    ),
                    analysis=FrameAnalysis(
                        semantic_segmentation=[],
                        object_detection=[],
                        face_recognition=[],
                    ),
                    analysis_artifacts=AnalysisArtifacts(
                        json="https://example.com/jobs/abc/analysis/json/frame_0.json?sig=1",
                    ),
                    metadata={
                        "provenance": {
                            "job_id": "abc-123",
                            "scene_id": None,
                            "frame_id": 0,
                            "timestamp": "00:00:05.000",
                            "raw_frame_index": 125,
                            "source_artifact_key": "https://example.com/jobs/abc/frames/original/frame_0.jpg?sig=1",
                        },
                        "model_provenance": [],
                        "evidence_anchors": [],
                    },
                )
            ],
        )
        assert result.job_id == "abc-123"
        assert len(result.frames) == 1
        assert result.frames[0].frame_id == 0

    def test_job_result_only_contains_cv_fields(self):
        result = JobResult(
            job_id="abc-123",
            frames=[],
        )
        payload = result.model_dump()
        assert payload["job_id"] == "abc-123"
        assert payload["frames"] == []
        assert payload["pipeline"] == {
            "stages": [],
            "status": [],
            "failed_stage": None,
            "mode": None,
        }
        assert payload["branch_metadata"] is None
        assert payload["video_face_identities"] is None
        assert payload["video_object_tracks"] is None
        assert payload["video_person_tracks"] is None
        assert payload["video_chunked_tracks"] is None

    def test_job_result_accepts_identity_tracking_payloads(self):
        result = JobResult(
            job_id="abc-123",
            frames=[],
            video_face_identities={
                "enabled": True,
                "model_id": "buffalo_l",
                "backend": "arcface",
                "provider_path": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                "active_provider": "CoreMLExecutionProvider",
                "scene_identities": [],
                "video_identities": [],
            },
            video_object_tracks={
                "enabled": True,
                "method": "object_tracking_v1",
                "tracks": [],
            },
            video_person_tracks={
                "enabled": True,
                "method": "object_face_fusion_v1",
                "tracks": [],
            },
        )

        payload = result.model_dump()
        assert payload["video_face_identities"]["model_id"] == "buffalo_l"
        assert payload["video_object_tracks"]["method"] == "object_tracking_v1"
        assert payload["video_person_tracks"]["method"] == "object_face_fusion_v1"

    def test_job_result_accepts_video_level_chunked_tracking_payload(self):
        result = JobResult(
            job_id="abc-123",
            frames=[],
            branch_metadata={
                "frame_analysis": {"status": "success"},
                "chunk_tracking": {"status": "success"},
            },
            video_chunked_tracks={
                "enabled": True,
                "method": "chunked_botsort_stitch_v1",
                "output_mode": "summary_v2",
                "tracks": [],
                "scenes": [],
                "zone_definition": {
                    "layout": "3x3",
                    "frame_width": 1920,
                    "frame_height": 1080,
                    "labels": [
                        "top-left",
                        "top-center",
                        "top-right",
                        "middle-left",
                        "center",
                        "middle-right",
                        "bottom-left",
                        "bottom-center",
                        "bottom-right",
                    ],
                    "zones": {
                        "top-left": {"x1": 0, "y1": 0, "x2": 640, "y2": 360},
                        "top-center": {"x1": 640, "y1": 0, "x2": 1280, "y2": 360},
                        "top-right": {"x1": 1280, "y1": 0, "x2": 1920, "y2": 360},
                        "middle-left": {"x1": 0, "y1": 360, "x2": 640, "y2": 720},
                        "center": {"x1": 640, "y1": 360, "x2": 1280, "y2": 720},
                        "middle-right": {"x1": 1280, "y1": 360, "x2": 1920, "y2": 720},
                        "bottom-left": {"x1": 0, "y1": 720, "x2": 640, "y2": 1080},
                        "bottom-center": {"x1": 640, "y1": 720, "x2": 1280, "y2": 1080},
                        "bottom-right": {"x1": 1280, "y1": 720, "x2": 1920, "y2": 1080},
                    },
                },
                "entities": [
                    {
                        "entity_id": "person-41",
                        "global_track_id": 41,
                        "entity_type": "person",
                        "label": "person",
                        "first_seen_ms": 601120,
                        "last_seen_ms": 617900,
                        "appearance_ranges_ms": [
                            {"start_ms": 601120, "end_ms": 617900}
                        ],
                        "zones_visited": ["top-left", "center", "bottom-right"],
                        "zone_occupancy": {
                            "bottom-right": 2,
                            "center": 6,
                            "top-left": 3,
                        },
                        "zone_transitions": [
                            {"from": "top-left", "to": "center", "at_ms": 606000},
                            {"from": "center", "to": "bottom-right", "at_ms": 612400},
                        ],
                        "evidence_timestamps_ms": [605800, 617900],
                    }
                ],
            },
        )

        payload = result.model_dump()
        assert payload["branch_metadata"]["frame_analysis"]["status"] == "success"
        entity = payload["video_chunked_tracks"]["entities"][0]
        assert entity["entity_id"] == "person-41"
        assert entity["zones_visited"] == ["top-left", "center", "bottom-right"]
        assert entity["zone_transitions"][0]["from"] == "top-left"
        assert entity["zone_transitions"][1]["to"] == "bottom-right"

    def test_job_result_accepts_legacy_compact_chunked_tracking_payload_for_rollback(self):
        result = JobResult(
            job_id="abc-123",
            frames=[],
            video_chunked_tracks={
                "enabled": True,
                "method": "chunked_botsort_stitch_v1",
                "output_mode": "legacy",
                "tracks": [
                    {
                        "id": 1,
                        "cls": 0,
                        "conf": 0.62,
                        "spans": [
                            {
                                "s_ms": 120340,
                                "e_ms": 127900,
                                "k": [[120340, 5123, 6011, 1234, 2200]],
                            }
                        ],
                    }
                ],
                "scenes": [
                    {
                        "scene_ts_ms": 120000,
                        "scene_te_ms": 128000,
                        "track_ids": [1],
                        "track_slices": [
                            {
                                "id": 1,
                                "cls": 0,
                                "k": [[120340, 5123, 6011, 1234, 2200]],
                            }
                        ],
                    }
                ],
            },
        )

        payload = result.model_dump()
        assert payload["video_chunked_tracks"]["output_mode"] == "legacy"
        assert payload["video_chunked_tracks"]["tracks"][0]["id"] == 1

    def test_job_result_rejects_invalid_entity_appearance_range(self):
        with pytest.raises(ValidationError):
            JobResult(
                job_id="abc-123",
                frames=[],
                video_chunked_tracks={
                    "enabled": True,
                    "method": "chunked_botsort_stitch_v1",
                    "output_mode": "summary_v2",
                    "tracks": [],
                    "scenes": [],
                    "zone_definition": {
                        "layout": "3x3",
                        "frame_width": 1000,
                        "frame_height": 1000,
                        "labels": ["top-left"],
                        "zones": {"top-left": {"x1": 0, "y1": 0, "x2": 333, "y2": 333}},
                    },
                    "entities": [
                        {
                            "entity_id": "person-1",
                            "global_track_id": 1,
                            "entity_type": "person",
                            "label": "person",
                            "first_seen_ms": 1000,
                            "last_seen_ms": 2000,
                            "appearance_ranges_ms": [
                                {
                                    "start_ms": 1000,
                                    "end_ms": "bad",
                                }
                            ],
                            "zones_visited": [],
                            "zone_occupancy": {},
                        }
                    ],
                },
            )


class TestCorpusSchemaValidation:
    def test_scene_entity_requires_evidence(self):
        with pytest.raises(ValidationError):
            SceneEntity(
                entity_id="entity_1",
                label="person",
                entity_type="object",
                count=1,
                confidence=0.9,
                temporal_span=SceneTemporalSpan(
                    first_seen=0.0, last_seen=1.0, duration_sec=1.0
                ),
                evidence=[],
            )
