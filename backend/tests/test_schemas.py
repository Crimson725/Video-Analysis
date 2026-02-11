"""Tests for app.schemas — Pydantic model validation and serialization."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    DetectionItem,
    FaceItem,
    FrameAnalysis,
    FrameFiles,
    FrameResult,
    JobResult,
    SegmentationItem,
)


class TestSegmentationItem:
    def test_class_alias_serialization(self):
        item = SegmentationItem(object_id=1, class_name="person", mask_polygon=[[0, 0], [1, 1]])
        data = item.model_dump(by_alias=True)
        assert "class" in data
        assert data["class"] == "person"
        assert "class_name" not in data

    def test_populate_by_name(self):
        item = SegmentationItem(object_id=1, class_name="car", mask_polygon=[[0, 0]])
        assert item.class_name == "car"


class TestDetectionItem:
    def test_valid_box(self):
        item = DetectionItem(label="dog", confidence=0.95, box=[10, 20, 30, 40])
        assert item.box == [10, 20, 30, 40]

    def test_rejects_box_with_3_elements(self):
        with pytest.raises(ValidationError):
            DetectionItem(label="dog", confidence=0.95, box=[10, 20, 30])

    def test_rejects_box_with_5_elements(self):
        with pytest.raises(ValidationError):
            DetectionItem(label="dog", confidence=0.95, box=[10, 20, 30, 40, 50])


class TestFaceItem:
    def test_valid_coordinates(self):
        item = FaceItem(face_id=1, confidence=0.99, coordinates=[10, 20, 30, 40])
        assert item.coordinates == [10, 20, 30, 40]

    def test_rejects_coordinates_with_5_elements(self):
        with pytest.raises(ValidationError):
            FaceItem(face_id=1, confidence=0.99, coordinates=[10, 20, 30, 40, 50])

    def test_rejects_coordinates_with_3_elements(self):
        with pytest.raises(ValidationError):
            FaceItem(face_id=1, confidence=0.99, coordinates=[10, 20, 30])


class TestJobResult:
    def test_valid_construction(self):
        result = JobResult(
            job_id="abc-123",
            frames=[
                FrameResult(
                    frame_id=0,
                    timestamp="00:00:05.000",
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
                )
            ],
        )
        assert result.job_id == "abc-123"
        assert len(result.frames) == 1
        assert result.frames[0].frame_id == 0
