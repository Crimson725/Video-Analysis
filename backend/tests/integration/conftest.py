"""Shared fixtures for integration tests using real models and real video."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.config import Settings
from app.models import ModelLoader
from app.scene import detect_scenes, extract_keyframes
from app.storage import MediaStoreError, R2MediaStore


# ---------------------------------------------------------------------------
# Resolve the test video path relative to the repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_VIDEO = _REPO_ROOT / "Test Videos" / "WhatCarCanYouGetForAGrand.mp4"
_R2_ENV_FILE = _REPO_ROOT / "backend" / ".env.r2"


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE env vars from a dotenv-style file."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


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


@pytest.fixture(scope="session")
def r2_store():
    """Return a real R2 media store; missing credentials are a hard failure."""
    _load_env_file(_R2_ENV_FILE)
    settings = Settings.from_env()
    missing = settings.missing_r2_fields()
    if missing:
        raise RuntimeError(f"Missing R2 settings for integration tests: {', '.join(missing)}")

    return R2MediaStore(
        account_id=settings.r2_account_id,
        bucket=settings.r2_bucket,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        default_url_ttl_seconds=settings.r2_url_ttl_seconds,
    )


@pytest.fixture()
def r2_cleanup_keys(r2_store):
    """Collect created R2 keys and delete them after the test."""
    created_keys: list[str] = []
    yield created_keys

    for object_key in created_keys:
        try:
            r2_store.delete_object(object_key)
        except MediaStoreError:
            # Best effort cleanup; failures are reported by read assertions in tests.
            pass


@pytest.fixture()
def r2_test_job_id():
    """Return a unique job id for R2 integration artifact tests."""
    return f"r2-itest-{uuid4().hex}"
