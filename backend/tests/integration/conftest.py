"""Shared fixtures for integration tests using real models and real video."""

from pathlib import Path

import numpy as np
import pytest

from app.models import ModelLoader
from app.scene import detect_scenes, extract_keyframes


# ---------------------------------------------------------------------------
# Resolve the test video path relative to the repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_VIDEO = _REPO_ROOT / "Test Videos" / "WhatCarCanYouGetForAGrand.mp4"


# ---------------------------------------------------------------------------
# 2.1 — Test video path fixture (session-scoped, skips if missing)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_video_path():
    """Return the absolute path to the test video, or skip if not present."""
    if not _TEST_VIDEO.is_file():
        pytest.skip(f"Test video not found: {_TEST_VIDEO}")
    return str(_TEST_VIDEO)


# ---------------------------------------------------------------------------
# 2.2 — Real model loader fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def models():
    """Load real YOLO and MTCNN models once per session."""
    return ModelLoader.get()


# ---------------------------------------------------------------------------
# 2.3 — Scene boundaries fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def scenes(test_video_path):
    """Detect scenes from the real test video once per session."""
    return detect_scenes(test_video_path)


# ---------------------------------------------------------------------------
# 2.4 — Keyframes fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def keyframes(test_video_path, scenes):
    """Extract keyframes from the real test video once per session."""
    frames = extract_keyframes(test_video_path, scenes)
    assert len(frames) > 0, "No keyframes extracted from test video"
    return frames


# ---------------------------------------------------------------------------
# 2.5 — Single sample frame fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_frame(keyframes):
    """Return the first extracted keyframe for use in analysis tests."""
    return keyframes[0]


# ---------------------------------------------------------------------------
# 2.6 — Temporary static directory fixture (function-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture()
def static_dir(tmp_path):
    """Return a fresh temporary directory usable as `static_dir`."""
    return str(tmp_path)
