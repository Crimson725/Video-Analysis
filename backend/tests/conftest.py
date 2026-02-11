"""Shared fixtures and mock factories for the video analysis test suite."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app import jobs


# ---------------------------------------------------------------------------
# 2.1 — Autouse fixture: clear job store before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_jobs():
    """Ensure every test starts with an empty job store."""
    jobs.jobs.clear()
    yield
    jobs.jobs.clear()


# ---------------------------------------------------------------------------
# 2.2 — Mock model loader fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_models():
    """Return an object with `.detector`, `.segmenter`, `.face_detector` MagicMock attributes."""
    models = SimpleNamespace(
        detector=MagicMock(name="detector"),
        segmenter=MagicMock(name="segmenter"),
        face_detector=MagicMock(name="face_detector"),
    )
    return models


# ---------------------------------------------------------------------------
# 2.3 — YOLO result mock factory
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_yolo_result():
    """Factory fixture that creates mock YOLO result objects.

    Parameters
    ----------
    boxes : list[list[float]] | None
        Each inner list is [x1, y1, x2, y2].
    cls_ids : list[int] | None
        Class IDs matching each box.
    confs : list[float] | None
        Confidence scores matching each box.
    names : dict[int, str] | None
        Class-ID-to-name mapping.
    mask_polygons : list[np.ndarray] | None
        If provided, the result includes masks (segmentation mode).
    """

    def _factory(
        *,
        boxes: list[list[float]] | None = None,
        cls_ids: list[int] | None = None,
        confs: list[float] | None = None,
        names: dict[int, str] | None = None,
        mask_polygons: list[np.ndarray] | None = None,
    ) -> MagicMock:
        result = MagicMock(name="yolo_result")

        # names mapping
        result.names = names or {}

        # plot() returns a dummy image
        result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        if boxes is not None:
            box_mock = MagicMock(name="boxes")
            box_mock.xyxy.cpu.return_value.numpy.return_value = np.array(boxes, dtype=np.float32)
            box_mock.conf.cpu.return_value.numpy.return_value = np.array(confs or [0.9] * len(boxes), dtype=np.float32)
            box_mock.cls.cpu.return_value.numpy.return_value = np.array(cls_ids or [0] * len(boxes), dtype=np.float32)
            result.boxes = box_mock
        else:
            result.boxes = None

        if mask_polygons is not None:
            mask_mock = MagicMock(name="masks")
            mask_mock.xy = mask_polygons
            result.masks = mask_mock
        else:
            result.masks = None

        return result

    return _factory


# ---------------------------------------------------------------------------
# 2.4 — Temporary directory fixture for file I/O tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def static_dir(tmp_path):
    """Return a fresh temporary directory usable as `static_dir`."""
    return str(tmp_path)
