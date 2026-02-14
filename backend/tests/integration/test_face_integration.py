"""Integration tests for face detection with real MTCNN model."""

from pathlib import Path

import pytest

from app.analysis import run_face_recognition


pytestmark = pytest.mark.integration

JOB_ID = "test-face-job"


# ---------------------------------------------------------------------------
# 6.2 — run_face_recognition returns structurally valid output
# ---------------------------------------------------------------------------

class TestRunFaceRecognitionIntegration:
    def test_returns_list(self, sample_frame, models, static_dir):
        items = run_face_recognition(
            sample_frame["image"],
            models.face_detector,
            JOB_ID,
            sample_frame["frame_id"],
            static_dir,
        )

        assert isinstance(items, list)

    def test_each_item_has_valid_structure(self, sample_frame, models, static_dir):
        items = run_face_recognition(
            sample_frame["image"],
            models.face_detector,
            JOB_ID,
            sample_frame["frame_id"],
            static_dir,
        )

        for item in items:
            # face_id: int > 0
            assert isinstance(item["face_id"], int)
            assert item["face_id"] > 0

            # identity_id: stable anonymous face identifier
            assert isinstance(item["identity_id"], str)
            assert item["identity_id"].startswith("face_")

            # confidence: float in [0, 1]
            assert isinstance(item["confidence"], float)
            assert 0.0 <= item["confidence"] <= 1.0

            # coordinates: list of exactly 4 ints
            assert isinstance(item["coordinates"], list)
            assert len(item["coordinates"]) == 4
            for coord in item["coordinates"]:
                assert isinstance(coord, int)

            # palette_rgb/bbox_rgb: RGB triplets
            assert isinstance(item["palette_rgb"], list)
            assert isinstance(item["bbox_rgb"], list)
            assert len(item["palette_rgb"]) == 3
            assert len(item["bbox_rgb"]) == 3
            for channel in item["palette_rgb"] + item["bbox_rgb"]:
                assert isinstance(channel, int)
                assert 0 <= channel <= 255


# ---------------------------------------------------------------------------
# 6.3 — run_face_recognition saves a visualization file
# ---------------------------------------------------------------------------

class TestFaceVisualization:
    def test_saves_visualization_jpeg(self, sample_frame, models, static_dir):
        frame_id = sample_frame["frame_id"]
        run_face_recognition(
            sample_frame["image"],
            models.face_detector,
            JOB_ID,
            frame_id,
            static_dir,
        )

        vis_path = Path(static_dir) / JOB_ID / "face" / f"frame_{frame_id}.jpg"
        assert vis_path.is_file(), f"Visualization file not found: {vis_path}"
        assert vis_path.stat().st_size > 0, "Visualization file is empty"
