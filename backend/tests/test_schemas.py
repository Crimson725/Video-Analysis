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
            person_track_id="person_track_123",
            person_identity_id="video_person_1",
            person_identity_source="video_person_id",
            person_identity_confidence=0.91,
        )

        assert item.object_track_id == "object_track_abc"
        assert item.person_track_id == "person_track_123"
        assert item.person_identity_id == "video_person_1"
        assert item.person_identity_source == "video_person_id"
        assert item.person_identity_confidence == pytest.approx(0.91)


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
        assert result.model_dump() == {
            "job_id": "abc-123",
            "pipeline": {
                "stages": [],
                "status": [],
                "failed_stage": None,
                "mode": None,
            },
            "frames": [],
        }


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
