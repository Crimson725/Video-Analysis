"""Integration tests for object detection with real YOLO model."""

from pathlib import Path

import pytest

from app.analysis import run_detection


pytestmark = pytest.mark.integration

JOB_ID = "test-det-job"


# ---------------------------------------------------------------------------
# 5.2 — run_detection returns structurally valid output
# ---------------------------------------------------------------------------

class TestRunDetectionIntegration:
    def test_returns_list(self, sample_frame, models, static_dir):
        items = run_detection(
            sample_frame["image"],
            models.detector,
            JOB_ID,
            sample_frame["frame_id"],
            static_dir,
        )

        assert isinstance(items, list)

    def test_each_item_has_valid_structure(self, sample_frame, models, static_dir):
        items = run_detection(
            sample_frame["image"],
            models.detector,
            JOB_ID,
            sample_frame["frame_id"],
            static_dir,
        )

        for item in items:
            # label: non-empty string
            assert isinstance(item["label"], str)
            assert len(item["label"]) > 0

            # confidence: float in [0, 1]
            assert isinstance(item["confidence"], float)
            assert 0.0 <= item["confidence"] <= 1.0

            # box: list of exactly 4 non-negative ints
            assert isinstance(item["box"], list)
            assert len(item["box"]) == 4
            for coord in item["box"]:
                assert isinstance(coord, int)
                assert coord >= 0


# ---------------------------------------------------------------------------
# 5.3 — run_detection saves a visualization file
# ---------------------------------------------------------------------------

class TestDetectionVisualization:
    def test_saves_visualization_jpeg(self, sample_frame, models, static_dir):
        frame_id = sample_frame["frame_id"]
        run_detection(
            sample_frame["image"],
            models.detector,
            JOB_ID,
            frame_id,
            static_dir,
        )

        vis_path = Path(static_dir) / JOB_ID / "det" / f"frame_{frame_id}.jpg"
        assert vis_path.is_file(), f"Visualization file not found: {vis_path}"
        assert vis_path.stat().st_size > 0, "Visualization file is empty"
