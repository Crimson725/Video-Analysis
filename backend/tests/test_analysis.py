"""Tests for app.analysis — frame analysis pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.analysis import (
    _to_int_coords,
    analyze_frame,
    run_detection,
    run_face_recognition,
    run_segmentation,
)


# ---------------------------------------------------------------------------
# 4.1 — _to_int_coords
# ---------------------------------------------------------------------------

class TestToIntCoords:
    def test_positive_float_rounds_up(self):
        assert _to_int_coords(10.6) == 11

    def test_positive_float_rounds_down(self):
        assert _to_int_coords(10.4) == 10

    def test_negative_float_rounds_to_zero(self):
        assert _to_int_coords(-0.4) == 0

    def test_negative_float_rounds_down(self):
        assert _to_int_coords(-1.6) == -2

    def test_exact_integer(self):
        assert _to_int_coords(5.0) == 5


# ---------------------------------------------------------------------------
# 4.2 — run_segmentation
# ---------------------------------------------------------------------------

class TestRunSegmentation:
    @patch("app.analysis.cv2.imwrite")
    def test_detected_objects_return_structured_items(
        self, mock_imwrite, make_yolo_result, static_dir
    ):
        yolo_result = make_yolo_result(
            boxes=[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
            cls_ids=[0, 1],
            confs=[0.9, 0.85],
            names={0: "person", 1: "car"},
            mask_polygons=[
                np.array([[10.5, 20.5], [30.5, 40.5]]),
                np.array([[50.5, 60.5], [70.5, 80.5]]),
            ],
        )
        model = MagicMock()
        model.return_value = [yolo_result]

        items = run_segmentation(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model,
            "job-1",
            0,
            static_dir,
        )

        assert len(items) == 2
        assert items[0]["object_id"] == 1
        assert items[0]["class"] == "person"
        assert isinstance(items[0]["mask_polygon"], list)
        assert items[1]["object_id"] == 2
        assert items[1]["class"] == "car"

    @patch("app.analysis.cv2.imwrite")
    def test_no_masks_returns_empty_list(
        self, mock_imwrite, make_yolo_result, static_dir
    ):
        yolo_result = make_yolo_result(boxes=None, mask_polygons=None)
        model = MagicMock()
        model.return_value = [yolo_result]

        items = run_segmentation(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model,
            "job-1",
            0,
            static_dir,
        )

        assert items == []

    @patch("app.analysis.cv2.imwrite")
    def test_visualization_saved(self, mock_imwrite, make_yolo_result, static_dir):
        yolo_result = make_yolo_result(
            boxes=[[10.0, 20.0, 30.0, 40.0]],
            cls_ids=[0],
            names={0: "person"},
            mask_polygons=[np.array([[10.0, 20.0]])],
        )
        model = MagicMock()
        model.return_value = [yolo_result]

        run_segmentation(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model,
            "job-1",
            0,
            static_dir,
        )

        mock_imwrite.assert_called_once()
        path = mock_imwrite.call_args[0][0]
        assert "seg" in path
        assert "frame_0.jpg" in path


# ---------------------------------------------------------------------------
# 4.3 — run_detection
# ---------------------------------------------------------------------------

class TestRunDetection:
    @patch("app.analysis.cv2.imwrite")
    def test_detected_boxes_return_structured_items(
        self, mock_imwrite, make_yolo_result, static_dir
    ):
        yolo_result = make_yolo_result(
            boxes=[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0], [1.0, 2.0, 3.0, 4.0]],
            cls_ids=[0, 1, 2],
            confs=[0.95, 0.88, 0.76],
            names={0: "person", 1: "car", 2: "dog"},
        )
        model = MagicMock()
        model.return_value = [yolo_result]

        items = run_detection(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model,
            "job-1",
            0,
            static_dir,
        )

        assert len(items) == 3
        assert items[0]["label"] == "person"
        assert items[0]["confidence"] == pytest.approx(0.95, abs=1e-5)
        assert items[0]["box"] == [10, 20, 30, 40]
        assert items[1]["label"] == "car"
        assert items[2]["label"] == "dog"

    @patch("app.analysis.cv2.imwrite")
    def test_no_boxes_returns_empty_list(
        self, mock_imwrite, make_yolo_result, static_dir
    ):
        yolo_result = make_yolo_result(boxes=None)
        model = MagicMock()
        model.return_value = [yolo_result]

        items = run_detection(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model,
            "job-1",
            0,
            static_dir,
        )

        assert items == []

    @patch("app.analysis.cv2.imwrite")
    def test_visualization_saved(self, mock_imwrite, make_yolo_result, static_dir):
        yolo_result = make_yolo_result(
            boxes=[[10.0, 20.0, 30.0, 40.0]],
            cls_ids=[0],
            names={0: "person"},
        )
        model = MagicMock()
        model.return_value = [yolo_result]

        run_detection(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model,
            "job-1",
            0,
            static_dir,
        )

        mock_imwrite.assert_called_once()
        path = mock_imwrite.call_args[0][0]
        assert "det" in path
        assert "frame_0.jpg" in path


# ---------------------------------------------------------------------------
# 4.4 — run_face_recognition
# ---------------------------------------------------------------------------

class TestRunFaceRecognition:
    @patch("app.analysis.cv2.imwrite")
    @patch("app.analysis.cv2.cvtColor")
    @patch("app.analysis.cv2.rectangle")
    def test_faces_above_threshold(
        self, mock_rect, mock_cvt, mock_imwrite, static_dir
    ):
        mock_cvt.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        face_detector = MagicMock()
        face_detector.detect.return_value = (
            np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
            np.array([0.95, 0.99]),
        )

        items = run_face_recognition(
            np.zeros((100, 100, 3), dtype=np.uint8),
            face_detector,
            "job-1",
            0,
            static_dir,
            confidence_threshold=0.9,
        )

        assert len(items) == 2
        assert items[0]["face_id"] == 1
        assert items[0]["confidence"] == 0.95
        assert items[0]["coordinates"] == [10, 20, 30, 40]
        assert items[1]["face_id"] == 2

    @patch("app.analysis.cv2.imwrite")
    @patch("app.analysis.cv2.cvtColor")
    @patch("app.analysis.cv2.rectangle")
    def test_faces_below_threshold_excluded(
        self, mock_rect, mock_cvt, mock_imwrite, static_dir
    ):
        mock_cvt.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        face_detector = MagicMock()
        face_detector.detect.return_value = (
            np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
            np.array([0.5, 0.95]),
        )

        items = run_face_recognition(
            np.zeros((100, 100, 3), dtype=np.uint8),
            face_detector,
            "job-1",
            0,
            static_dir,
            confidence_threshold=0.9,
        )

        assert len(items) == 1
        assert items[0]["confidence"] == 0.95

    @patch("app.analysis.cv2.imwrite")
    @patch("app.analysis.cv2.cvtColor")
    def test_no_faces_returns_empty_list(
        self, mock_cvt, mock_imwrite, static_dir
    ):
        mock_cvt.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        face_detector = MagicMock()
        face_detector.detect.return_value = (None, None)

        items = run_face_recognition(
            np.zeros((100, 100, 3), dtype=np.uint8),
            face_detector,
            "job-1",
            0,
            static_dir,
        )

        assert items == []

    @patch("app.analysis.cv2.imwrite")
    @patch("app.analysis.cv2.cvtColor")
    @patch("app.analysis.cv2.rectangle")
    def test_visualization_saved_when_faces_found(
        self, mock_rect, mock_cvt, mock_imwrite, static_dir
    ):
        mock_cvt.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        face_detector = MagicMock()
        face_detector.detect.return_value = (
            np.array([[10.0, 20.0, 30.0, 40.0]]),
            np.array([0.95]),
        )

        run_face_recognition(
            np.zeros((100, 100, 3), dtype=np.uint8),
            face_detector,
            "job-1",
            0,
            static_dir,
        )

        mock_imwrite.assert_called_once()
        path = mock_imwrite.call_args[0][0]
        assert "face" in path
        assert "frame_0.jpg" in path

    @patch("app.analysis.cv2.imwrite")
    @patch("app.analysis.cv2.cvtColor")
    def test_visualization_saved_when_no_faces(
        self, mock_cvt, mock_imwrite, static_dir
    ):
        mock_cvt.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        face_detector = MagicMock()
        face_detector.detect.return_value = (None, None)

        run_face_recognition(
            np.zeros((100, 100, 3), dtype=np.uint8),
            face_detector,
            "job-1",
            0,
            static_dir,
        )

        mock_imwrite.assert_called_once()
        path = mock_imwrite.call_args[0][0]
        assert "face" in path


# ---------------------------------------------------------------------------
# 4.5 — analyze_frame
# ---------------------------------------------------------------------------

class TestAnalyzeFrame:
    @patch("app.analysis.run_face_recognition")
    @patch("app.analysis.run_detection")
    @patch("app.analysis.run_segmentation")
    def test_orchestrates_all_pipelines(
        self, mock_seg, mock_det, mock_face, mock_models, static_dir
    ):
        mock_seg.return_value = [{"object_id": 1, "class": "person", "mask_polygon": [[0, 0]]}]
        mock_det.return_value = [{"label": "car", "confidence": 0.9, "box": [1, 2, 3, 4]}]
        mock_face.return_value = [{"face_id": 1, "confidence": 0.95, "coordinates": [5, 6, 7, 8]}]

        frame_data = {
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "frame_id": 0,
            "timestamp": "00:00:05.000",
        }

        result = analyze_frame(frame_data, mock_models, "job-1", static_dir)

        assert result["frame_id"] == 0
        assert result["timestamp"] == "00:00:05.000"
        assert "files" in result
        assert "original" in result["files"]
        assert "segmentation" in result["files"]
        assert "detection" in result["files"]
        assert "face" in result["files"]
        assert "analysis" in result
        assert "semantic_segmentation" in result["analysis"]
        assert "object_detection" in result["analysis"]
        assert "face_recognition" in result["analysis"]
        assert len(result["analysis"]["semantic_segmentation"]) == 1
        assert len(result["analysis"]["object_detection"]) == 1
        assert len(result["analysis"]["face_recognition"]) == 1

        mock_seg.assert_called_once()
        mock_det.assert_called_once()
        mock_face.assert_called_once()
