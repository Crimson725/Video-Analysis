"""Tests for app.scene — scene detection and keyframe extraction."""

from unittest.mock import MagicMock, patch

import numpy as np

from app.scene import detect_scenes, extract_keyframes, save_original_frames


class TestDetectScenes:
    @patch("app.scene.detect")
    def test_returns_scene_tuples(self, mock_detect):
        # Each scene boundary is a (start_timecode, end_timecode) pair
        tc1_start = MagicMock()
        tc1_start.get_seconds.return_value = 0.0
        tc1_end = MagicMock()
        tc1_end.get_seconds.return_value = 5.0

        tc2_start = MagicMock()
        tc2_start.get_seconds.return_value = 5.0
        tc2_end = MagicMock()
        tc2_end.get_seconds.return_value = 12.0

        tc3_start = MagicMock()
        tc3_start.get_seconds.return_value = 12.0
        tc3_end = MagicMock()
        tc3_end.get_seconds.return_value = 20.0

        mock_detect.return_value = [
            (tc1_start, tc1_end),
            (tc2_start, tc2_end),
            (tc3_start, tc3_end),
        ]

        result = detect_scenes("fake_video.mp4")

        assert len(result) == 3
        assert result[0] == (0.0, 5.0)
        assert result[1] == (5.0, 12.0)
        assert result[2] == (12.0, 20.0)

    @patch("app.scene.detect")
    def test_empty_scene_list(self, mock_detect):
        mock_detect.return_value = []
        result = detect_scenes("fake_video.mp4")
        assert result == []


class TestExtractKeyframes:
    @patch("app.scene.cv2")
    def test_extracts_keyframes_for_each_scene(self, mock_cv2):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.get.return_value = 25.0  # fps
        mock_cap.read.return_value = (True, fake_frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FPS = 5  # cv2.CAP_PROP_FPS constant
        mock_cv2.CAP_PROP_POS_FRAMES = 1  # cv2.CAP_PROP_POS_FRAMES constant

        scenes = [(0.0, 10.0), (10.0, 20.0)]
        result = extract_keyframes("fake_video.mp4", scenes)

        assert len(result) == 2
        for i, frame in enumerate(result):
            assert frame["frame_id"] == i
            assert "timestamp" in frame
            # Timestamp should be HH:MM:SS.mmm format
            parts = frame["timestamp"].split(":")
            assert len(parts) == 3
            assert isinstance(frame["image"], np.ndarray)

        mock_cap.release.assert_called_once()

    @patch("app.scene.cv2")
    def test_failed_read_is_skipped(self, mock_cv2):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.get.return_value = 25.0
        # First read fails, second succeeds
        mock_cap.read.side_effect = [(False, None), (True, fake_frame)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        scenes = [(0.0, 5.0), (5.0, 10.0)]
        result = extract_keyframes("fake_video.mp4", scenes)

        # Only 1 frame because the first read failed
        assert len(result) == 1
        assert result[0]["frame_id"] == 1

    @patch("app.scene.cv2")
    def test_timestamp_format(self, mock_cv2):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.get.return_value = 25.0
        mock_cap.read.return_value = (True, fake_frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        # Scene with mid-point at 3665.5 seconds = 1h 1m 5.5s
        scenes = [(3660.0, 3671.0)]
        result = extract_keyframes("fake_video.mp4", scenes)

        assert len(result) == 1
        ts = result[0]["timestamp"]
        assert ts.startswith("01:01:")


class TestSaveOriginalFrames:
    @patch("app.scene.cv2.imwrite")
    def test_saves_frames_to_correct_paths(self, mock_imwrite, static_dir):
        frames = [
            {"frame_id": 0, "image": np.zeros((100, 100, 3), dtype=np.uint8)},
            {"frame_id": 1, "image": np.zeros((100, 100, 3), dtype=np.uint8)},
        ]

        save_original_frames(frames, "job-123", static_dir)

        assert mock_imwrite.call_count == 2
        paths = [call.args[0] for call in mock_imwrite.call_args_list]
        assert any("frame_0.jpg" in p for p in paths)
        assert any("frame_1.jpg" in p for p in paths)
        assert all("original" in p for p in paths)
