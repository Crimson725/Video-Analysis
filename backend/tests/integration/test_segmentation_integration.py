"""Integration tests for semantic segmentation with real YOLO-seg model."""

from pathlib import Path

import pytest

from app.analysis import run_segmentation


pytestmark = pytest.mark.integration

JOB_ID = "test-seg-job"


# ---------------------------------------------------------------------------
# 4.2 — run_segmentation returns structurally valid output
# ---------------------------------------------------------------------------

class TestRunSegmentationIntegration:
    def test_returns_list(self, sample_frame, models, static_dir):
        items = run_segmentation(
            sample_frame["image"],
            models.segmenter,
            JOB_ID,
            sample_frame["frame_id"],
            static_dir,
        )

        assert isinstance(items, list)

    def test_each_item_has_valid_structure(self, sample_frame, models, static_dir):
        items = run_segmentation(
            sample_frame["image"],
            models.segmenter,
            JOB_ID,
            sample_frame["frame_id"],
            static_dir,
        )

        for item in items:
            # object_id: int > 0
            assert isinstance(item["object_id"], int)
            assert item["object_id"] > 0

            # class: non-empty string
            assert isinstance(item["class"], str)
            assert len(item["class"]) > 0

            # mask_polygon: list of [x, y] int pairs
            assert isinstance(item["mask_polygon"], list)
            for point in item["mask_polygon"]:
                assert isinstance(point, list)
                assert len(point) == 2
                assert isinstance(point[0], int)
                assert isinstance(point[1], int)

            # palette_rgb/bbox_rgb: RGB triplets
            assert isinstance(item["palette_rgb"], list)
            assert isinstance(item["bbox_rgb"], list)
            assert len(item["palette_rgb"]) == 3
            assert len(item["bbox_rgb"]) == 3
            for channel in item["palette_rgb"] + item["bbox_rgb"]:
                assert isinstance(channel, int)
                assert 0 <= channel <= 255


# ---------------------------------------------------------------------------
# 4.3 — run_segmentation saves a visualization file
# ---------------------------------------------------------------------------

class TestSegmentationVisualization:
    def test_saves_visualization_jpeg(self, sample_frame, models, static_dir):
        frame_id = sample_frame["frame_id"]
        run_segmentation(
            sample_frame["image"],
            models.segmenter,
            JOB_ID,
            frame_id,
            static_dir,
        )

        vis_path = Path(static_dir) / JOB_ID / "seg" / f"frame_{frame_id}.jpg"
        assert vis_path.is_file(), f"Visualization file not found: {vis_path}"
        assert vis_path.stat().st_size > 0, "Visualization file is empty"
