"""Integration tests for real R2 JSON artifact storage."""

import json

import pytest

from app.storage import MediaStoreError

pytestmark = pytest.mark.integration


def _sample_frame_payload(job_id: str, frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": "00:00:01.000",
        "files": {
            "original": f"jobs/{job_id}/frames/original/frame_{frame_id}.jpg",
            "segmentation": f"jobs/{job_id}/frames/seg/frame_{frame_id}.jpg",
            "detection": f"jobs/{job_id}/frames/det/frame_{frame_id}.jpg",
            "face": f"jobs/{job_id}/frames/face/frame_{frame_id}.jpg",
        },
        "analysis": {
            "semantic_segmentation": [],
            "object_detection": [],
            "face_recognition": [],
        },
        "analysis_artifacts": {
            "json": f"jobs/{job_id}/analysis/json/frame_{frame_id}.json",
        },
    }


class TestR2AnalysisArtifactsIntegration:
    def test_write_and_read_json_artifacts(self, r2_store, r2_test_job_id, r2_cleanup_keys):
        payload = _sample_frame_payload(r2_test_job_id, frame_id=0)
        json_bytes = json.dumps(payload).encode("utf-8")
        packet_bytes = b'{"scene":{"scene_id":0}}'

        json_key = r2_store.upload_analysis_artifact(r2_test_job_id, "json", 0, json_bytes)
        r2_cleanup_keys.append(json_key)
        packet_key = r2_store.upload_scene_artifact(r2_test_job_id, "packet", 0, packet_bytes)
        r2_cleanup_keys.append(packet_key)

        read_json = r2_store.read_object(json_key)
        read_packet = r2_store.read_object(packet_key)

        assert read_json == json_bytes
        assert read_packet == packet_bytes

    def test_cleanup_executes_in_failure_path(self, r2_store, r2_test_job_id):
        payload = _sample_frame_payload(r2_test_job_id, frame_id=1)
        json_bytes = json.dumps(payload).encode("utf-8")

        created_keys: list[str] = []
        json_key = ""
        packet_key = ""
        try:
            json_key = r2_store.upload_analysis_artifact(r2_test_job_id, "json", 1, json_bytes)
            created_keys.append(json_key)
            packet_key = r2_store.upload_scene_artifact(
                r2_test_job_id,
                "packet",
                1,
                b'{"scene":{"scene_id":1}}',
            )
            created_keys.append(packet_key)

            # Simulate verification failure after writes.
            raise RuntimeError("simulated verification failure")
        except RuntimeError:
            pass
        finally:
            for object_key in created_keys:
                r2_store.delete_object(object_key)

        with pytest.raises(MediaStoreError):
            r2_store.read_object(json_key)
        with pytest.raises(MediaStoreError):
            r2_store.read_object(packet_key)
