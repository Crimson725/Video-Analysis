"""Integration tests for the full analyze_frame pipeline with real models."""

from pathlib import Path

import pytest

from app.analysis import analyze_frame
from app.scene import save_original_frames


pytestmark = pytest.mark.integration

JOB_ID = "test-analyze-job"


# ---------------------------------------------------------------------------
# 7.2 — analyze_frame returns complete structured result
# ---------------------------------------------------------------------------

class TestAnalyzeFrameIntegration:
    def test_returns_dict_with_required_keys(self, sample_frame, models, static_dir):
        # Save original frame first (analyze_frame references it in file paths)
        save_original_frames([sample_frame], JOB_ID, static_dir)

        result = analyze_frame(sample_frame, models, JOB_ID, static_dir)

        assert isinstance(result, dict)

        # frame_id: int
        assert isinstance(result["frame_id"], int)

        # timestamp: str
        assert isinstance(result["timestamp"], str)

        # files: dict with all four keys, each a non-empty string containing job_id
        files = result["files"]
        assert isinstance(files, dict)
        for key in ("original", "segmentation", "detection", "face"):
            assert key in files, f"Missing files key: {key}"
            assert isinstance(files[key], str)
            assert len(files[key]) > 0
            assert JOB_ID in files[key], f"Job ID not in files['{key}']: {files[key]}"

        # analysis: dict with all three keys, each a list
        analysis = result["analysis"]
        assert isinstance(analysis, dict)
        for key in ("semantic_segmentation", "object_detection", "face_recognition"):
            assert key in analysis, f"Missing analysis key: {key}"
            assert isinstance(analysis[key], list)


# ---------------------------------------------------------------------------
# 7.3 — analyze_frame produces visualization files
# ---------------------------------------------------------------------------

class TestAnalyzeFrameVisualization:
    def test_produces_all_visualization_files(self, sample_frame, models, static_dir):
        save_original_frames([sample_frame], JOB_ID, static_dir)
        frame_id = sample_frame["frame_id"]

        analyze_frame(sample_frame, models, JOB_ID, static_dir)

        for subdir in ("seg", "det", "face"):
            vis_path = Path(static_dir) / JOB_ID / subdir / f"frame_{frame_id}.jpg"
            assert vis_path.is_file(), f"Visualization not found: {vis_path}"
            assert vis_path.stat().st_size > 0, f"Visualization empty: {vis_path}"
