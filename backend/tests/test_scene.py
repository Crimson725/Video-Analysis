"""Tests for app.scene — scene detection and keyframe extraction."""

from unittest.mock import MagicMock, patch

import numpy as np

from app.scene import (
    _ScoredCandidate,
    _default_scene_budget,
    _robust_change_threshold,
    _temporal_nms,
    detect_scenes,
    extract_keyframes,
    extract_tracking_frames,
    save_original_frames,
)


def _frame_with_intensity(intensity: int) -> np.ndarray:
    clipped = max(0, min(255, intensity))
    return np.full((120, 200, 3), clipped, dtype=np.uint8)


class _StubCapture:
    def __init__(self, *, fps: float, frames_by_index: dict[int, np.ndarray]):
        self._fps = fps
        self._frames_by_index = frames_by_index
        self._position = 0
        self.released = False

    def get(self, _prop: int) -> float:
        return self._fps

    def set(self, _prop: int, value: float) -> None:
        self._position = int(value)

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self._frames_by_index.get(self._position)
        if frame is None:
            return False, None
        return True, frame.copy()

    def release(self) -> None:
        self.released = True


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
    def test_default_scene_budget_tiers(self):
        assert _default_scene_budget(8.0) == 12
        assert _default_scene_budget(10.0) == 12
        assert _default_scene_budget(10.1) == 30
        assert _default_scene_budget(60.0) == 30
        assert _default_scene_budget(60.1) == 60

    def test_threshold_uses_floor_and_robust_stats(self):
        assert _robust_change_threshold([0.0, 0.0, 0.0]) == 0.03
        assert _robust_change_threshold([0.0, 0.1, 0.2]) > 0.03

    def test_temporal_nms_prefers_higher_score_in_window(self):
        candidates = [
            _ScoredCandidate(timestamp_sec=1.0, frame_index=10, score=0.10),
            _ScoredCandidate(timestamp_sec=1.4, frame_index=14, score=0.40),
            _ScoredCandidate(timestamp_sec=2.2, frame_index=22, score=0.25),
        ]

        kept = _temporal_nms(candidates, delta_sec=0.7)

        assert len(kept) == 2
        assert kept[0].timestamp_sec == 1.4
        assert kept[0].score == 0.40
        assert kept[1].timestamp_sec == 2.2

    @patch("app.scene.cv2.VideoCapture")
    def test_extracts_anchor_frames_for_static_scene(self, mock_video_capture):
        cap = _StubCapture(
            fps=10.0,
            frames_by_index={idx: _frame_with_intensity(0) for idx in range(0, 40)},
        )
        mock_video_capture.return_value = cap

        result = extract_keyframes("fake_video.mp4", [(0.0, 2.0)])

        assert [frame["frame_id"] for frame in result] == [0, 1, 2]
        assert [frame["scene_id"] for frame in result] == [0, 0, 0]
        assert [frame["timestamp"] for frame in result] == [
            "00:00:00.000",
            "00:00:01.000",
            "00:00:01.900",
        ]
        assert [frame["raw_frame_index"] for frame in result] == [0, 10, 19]
        assert cap.released is True

    @patch("app.scene.cv2.VideoCapture")
    def test_respects_default_budget_for_short_scene(self, mock_video_capture):
        # Intensities change every 1.0s while scan step is 0.5s -> alternating low/high diffs.
        cap = _StubCapture(
            fps=10.0,
            frames_by_index={
                idx: _frame_with_intensity((idx // 10) * 20) for idx in range(0, 130)
            },
        )
        mock_video_capture.return_value = cap

        result = extract_keyframes("fake_video.mp4", [(0.0, 10.0)])

        assert 3 <= len(result) <= 12  # Budget tier for <=10s scenes.
        assert [frame["frame_id"] for frame in result] == list(range(len(result)))
        timestamps = [frame["timestamp"] for frame in result]
        assert "00:00:00.000" in timestamps
        assert "00:00:05.000" in timestamps
        assert "00:00:09.900" in timestamps

    @patch("app.scene.cv2.VideoCapture")
    def test_frame_ids_are_sequential_and_scene_ids_are_preserved(
        self, mock_video_capture
    ):
        cap = _StubCapture(
            fps=10.0,
            frames_by_index={idx: _frame_with_intensity(0) for idx in range(0, 80)},
        )
        mock_video_capture.return_value = cap

        result = extract_keyframes("fake_video.mp4", [(0.0, 2.0), (2.0, 4.0)])

        assert len(result) == 6
        assert [frame["frame_id"] for frame in result] == [0, 1, 2, 3, 4, 5]
        assert [frame["scene_id"] for frame in result] == [0, 0, 0, 1, 1, 1]
        timestamps = [frame["timestamp"] for frame in result]
        assert timestamps == sorted(timestamps)

    @patch("app.scene.cv2.VideoCapture")
    def test_failed_reads_are_skipped_without_frame_id_gaps(self, mock_video_capture):
        frames_by_index = {idx: _frame_with_intensity(0) for idx in range(0, 40)}
        # For scene (0, 2) at 10 FPS, midpoint anchor maps to frame index 10.
        frames_by_index.pop(10)
        cap = _StubCapture(fps=10.0, frames_by_index=frames_by_index)
        mock_video_capture.return_value = cap

        result = extract_keyframes("fake_video.mp4", [(0.0, 2.0)])

        assert len(result) == 2
        assert [frame["frame_id"] for frame in result] == [0, 1]
        assert [frame["timestamp"] for frame in result] == ["00:00:00.000", "00:00:01.900"]
        assert [frame["raw_frame_index"] for frame in result] == [0, 19]

    @patch("app.scene.cv2.VideoCapture")
    def test_timestamp_format_for_large_hour_values(self, mock_video_capture):
        cap = _StubCapture(
            fps=25.0,
            frames_by_index={idx: _frame_with_intensity(0) for idx in range(0, 100000)},
        )
        mock_video_capture.return_value = cap

        result = extract_keyframes("fake_video.mp4", [(3660.0, 3671.0)])

        assert len(result) >= 1
        assert result[0]["timestamp"].startswith("01:01:")


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


class TestExtractTrackingFrames:
    @patch("app.scene.cv2")
    def test_extracts_deterministic_tracking_frames(self, mock_cv2):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.get.return_value = 10.0
        mock_cap.read.return_value = (True, fake_frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        scenes = [(0.0, 1.0), (2.0, 3.0)]
        result = extract_tracking_frames(
            "fake_video.mp4",
            scenes,
            sample_fps=2,
            max_samples_per_scene=3,
        )

        assert len(result) == 6
        assert result[0]["frame_id"] == 0
        assert result[1]["frame_id"] == 1
        assert result[2]["frame_id"] == 2
        assert result[3]["frame_id"] == 1_000_000
        assert result[0]["timestamp"] == "00:00:00.000"
        assert result[1]["timestamp"] == "00:00:00.500"
        assert result[2]["timestamp"] == "00:00:01.000"
        assert result[3]["timestamp"] == "00:00:02.000"
        assert all(frame["is_tracking_frame"] is True for frame in result)
        assert [frame["source_frame_index"] for frame in result[:4]] == [0, 5, 10, 20]
        mock_cap.release.assert_called_once()

    @patch("app.scene.cv2")
    def test_extract_tracking_frames_respects_max_samples(self, mock_cv2):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.get.return_value = 25.0
        mock_cap.read.return_value = (True, fake_frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        result = extract_tracking_frames(
            "fake_video.mp4",
            [(0.0, 10.0)],
            sample_fps=5,
            max_samples_per_scene=4,
        )

        assert len(result) == 4
        assert [frame["sample_index"] for frame in result] == [0, 1, 2, 3]
