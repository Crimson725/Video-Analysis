"""Integration tests for scene detection and keyframe extraction with real video."""

import re

import numpy as np
import pytest

from app.scene import detect_scenes, extract_keyframes


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 3.2 — detect_scenes returns valid scene boundaries
# ---------------------------------------------------------------------------

class TestDetectScenesIntegration:
    def test_returns_nonempty_scene_list(self, test_video_path):
        scenes = detect_scenes(test_video_path)

        assert isinstance(scenes, list)
        assert len(scenes) > 0

    def test_each_scene_has_valid_boundaries(self, test_video_path):
        scenes = detect_scenes(test_video_path)

        for start_sec, end_sec in scenes:
            assert isinstance(start_sec, (int, float))
            assert isinstance(end_sec, (int, float))
            assert start_sec >= 0, f"start_sec should be >= 0, got {start_sec}"
            assert end_sec > start_sec, (
                f"end_sec ({end_sec}) should be > start_sec ({start_sec})"
            )


# ---------------------------------------------------------------------------
# 3.3 — extract_keyframes returns valid frame dicts
# ---------------------------------------------------------------------------

class TestExtractKeyframesIntegration:
    def test_returns_nonempty_frame_list(self, test_video_path, scenes):
        frames = extract_keyframes(test_video_path, scenes)

        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_each_frame_has_required_keys(self, keyframes):
        for frame in keyframes:
            assert "frame_id" in frame
            assert "timestamp" in frame
            assert "image" in frame

    def test_frame_id_is_nonnegative_int(self, keyframes):
        for frame in keyframes:
            assert isinstance(frame["frame_id"], int)
            assert frame["frame_id"] >= 0

    def test_timestamp_matches_expected_format(self, keyframes):
        pattern = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}$")
        for frame in keyframes:
            assert isinstance(frame["timestamp"], str)
            assert pattern.match(frame["timestamp"]), (
                f"Timestamp '{frame['timestamp']}' does not match HH:MM:SS.mmm"
            )

    def test_image_is_valid_numpy_array(self, keyframes):
        for frame in keyframes:
            img = frame["image"]
            assert isinstance(img, np.ndarray)
            assert img.ndim == 3, f"Expected 3D array, got {img.ndim}D"
            assert img.dtype == np.uint8
            assert img.shape[0] > 0, "Image height should be > 0"
            assert img.shape[1] > 0, "Image width should be > 0"
