"""Tests for FastAPI API endpoints defined in app.main."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app import jobs


# ---------------------------------------------------------------------------
# 5.1 — Async client fixture with patched ModelLoader and scheduler
# ---------------------------------------------------------------------------

@pytest.fixture()
async def client():
    """Yield an httpx AsyncClient connected to the FastAPI app.

    ModelLoader.get and cleanup scheduler are patched to avoid real model
    loading and scheduler startup during tests.
    """
    with (
        patch("app.main.ModelLoader") as mock_model_loader,
        patch("app.main.cleanup") as mock_cleanup,
        patch("app.main.get_media_store") as mock_get_media_store,
    ):
        mock_model_loader.get.return_value = MagicMock(name="MockModelLoader")
        mock_cleanup.setup_scheduler = MagicMock()
        mock_cleanup.shutdown_scheduler = MagicMock()
        mock_store = MagicMock(name="MockMediaStore")
        mock_store.sign_read_url.side_effect = (
            lambda key, expires_in=None: f"https://signed.example/{key}?exp={expires_in or 3600}"
        )
        mock_get_media_store.return_value = mock_store

        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# 5.2 — POST /analyze-video happy path
# ---------------------------------------------------------------------------

class TestAnalyzeVideoHappyPath:
    async def test_valid_upload_returns_202_with_job_id(self, client):
        response = await client.post(
            "/analyze-video",
            files={"file": ("test_video.mp4", b"fake video content", "video/mp4")},
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0

    async def test_cleanup_override_false_is_persisted(self, client):
        response = await client.post(
            "/analyze-video",
            data={"cleanup_local_video_after_upload": "false"},
            files={"file": ("test_video.mp4", b"fake video content", "video/mp4")},
        )

        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = jobs.get_job(job_id)
        assert job is not None
        assert job["cleanup_local_video_after_upload"] is False


# ---------------------------------------------------------------------------
# 5.3 — POST /analyze-video error paths
# ---------------------------------------------------------------------------

class TestAnalyzeVideoErrors:
    async def test_empty_filename_returns_422(self, client):
        response = await client.post(
            "/analyze-video",
            files={"file": ("", b"content", "video/mp4")},
        )

        assert response.status_code == 422

    async def test_oversized_file_returns_413(self, client):
        # Create content slightly over the 500 MB limit
        # We patch MAX_UPLOAD_BYTES to a small value to keep the test fast
        with patch("app.main.MAX_UPLOAD_BYTES", 100):
            response = await client.post(
                "/analyze-video",
                files={"file": ("big.mp4", b"x" * 200, "video/mp4")},
            )

        assert response.status_code == 413


# ---------------------------------------------------------------------------
# 5.4 — GET /status/{job_id}
# ---------------------------------------------------------------------------

class TestGetStatus:
    async def test_processing_job(self, client):
        job_id = jobs.create_job()

        response = await client.get(f"/status/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "processing"
        assert data["error"] is None

    async def test_completed_job(self, client):
        job_id = jobs.create_job()
        jobs.complete_job(job_id, {"job_id": job_id, "frames": []})

        response = await client.get(f"/status/{job_id}")

        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    async def test_failed_job(self, client):
        job_id = jobs.create_job()
        jobs.fail_job(job_id, "some error")

        response = await client.get(f"/status/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "some error"

    async def test_unknown_job_returns_404(self, client):
        response = await client.get("/status/nonexistent-id")

        assert response.status_code == 404

    async def test_queue_stage_still_returns_processing_status(self, client):
        job_id = jobs.create_job(metadata={"stage": "waiting_scene_ai"})

        response = await client.get(f"/status/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["error"] is None


# ---------------------------------------------------------------------------
# 5.5 — GET /results/{job_id}
# ---------------------------------------------------------------------------

class TestGetResults:
    async def test_completed_job_returns_results(self, client):
        job_id = jobs.create_job()
        payload = {
            "job_id": job_id,
            "frames": [
                {
                    "frame_id": 0,
                    "timestamp": "00:00:05.000",
                    "files": {
                        "original": f"jobs/{job_id}/frames/original/frame_0.jpg",
                        "segmentation": f"jobs/{job_id}/frames/seg/frame_0.jpg",
                        "detection": f"jobs/{job_id}/frames/det/frame_0.jpg",
                        "face": f"jobs/{job_id}/frames/face/frame_0.jpg",
                    },
                    "analysis": {
                        "semantic_segmentation": [],
                        "object_detection": [],
                        "face_recognition": [],
                    },
                    "analysis_artifacts": {
                        "json": f"jobs/{job_id}/analysis/json/frame_0.json",
                    },
                    "metadata": {
                        "provenance": {
                            "job_id": job_id,
                            "scene_id": None,
                            "frame_id": 0,
                            "timestamp": "00:00:05.000",
                            "source_artifact_key": f"jobs/{job_id}/frames/original/frame_0.jpg",
                        },
                        "model_provenance": [],
                        "evidence_anchors": [],
                    },
                }
            ],
            "scene_narratives": [
                {
                    "scene_id": 0,
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "narrative_paragraph": "Scene summary.",
                    "key_moments": ["moment 1"],
                    "artifacts": {
                        "packet": f"jobs/{job_id}/scene/packets/scene_0.json",
                        "narrative": f"jobs/{job_id}/scene/narratives/scene_0.json",
                    },
                }
            ],
            "video_synopsis": {
                "synopsis": "Video synopsis.",
                "artifact": f"jobs/{job_id}/summary/synopsis.json",
                "model": "gemini-3-flash-preview",
            },
        }
        jobs.complete_job(job_id, payload)

        response = await client.get(f"/results/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert len(data["frames"]) == 1
        assert data["frames"][0]["files"]["original"].startswith("https://signed.example/jobs/")
        assert data["frames"][0]["analysis_artifacts"]["json"].startswith("https://signed.example/jobs/")
        assert "toon" not in data["frames"][0]["analysis_artifacts"]
        assert data["scene_narratives"][0]["artifacts"]["packet"].startswith("https://signed.example/jobs/")
        assert data["scene_narratives"][0]["artifacts"]["narrative"].startswith("https://signed.example/jobs/")
        assert data["video_synopsis"]["artifact"].startswith("https://signed.example/jobs/")

    async def test_processing_job_returns_409(self, client):
        job_id = jobs.create_job()

        response = await client.get(f"/results/{job_id}")

        assert response.status_code == 409

    async def test_failed_job_returns_409(self, client):
        job_id = jobs.create_job()
        jobs.fail_job(job_id, "error occurred")

        response = await client.get(f"/results/{job_id}")

        assert response.status_code == 409

    async def test_unknown_job_returns_404(self, client):
        response = await client.get("/results/nonexistent-id")

        assert response.status_code == 404

    async def test_queue_stage_results_contract_remains_409_while_processing(self, client):
        job_id = jobs.create_job(metadata={"stage": "scene_ai_processing"})

        response = await client.get(f"/results/{job_id}")

        assert response.status_code == 409
        assert response.json()["detail"] == "Job is still processing"
