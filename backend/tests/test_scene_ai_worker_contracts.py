"""Tests for scene AI worker queue payload contracts."""

from app.scene_ai_worker_contracts import SceneWorkerTaskInput


def test_task_input_round_trip_preserves_video_face_identities():
    payload = SceneWorkerTaskInput(
        job_id="job-1",
        scenes=[(0.0, 1.0)],
        frame_results=[{"frame_id": 0}],
        source_key="jobs/job-1/input/source.mp4",
        video_face_identities={
            "video_identities": [{"video_person_id": "video_person_1"}],
        },
    ).to_payload()

    restored = SceneWorkerTaskInput.from_payload(payload)

    assert restored.video_face_identities == {
        "video_identities": [{"video_person_id": "video_person_1"}],
    }


def test_task_input_invalid_video_face_identities_defaults_to_none():
    restored = SceneWorkerTaskInput.from_payload(
        {
            "job_id": "job-2",
            "scenes": [[0.0, 1.0]],
            "frame_results": [{"frame_id": 0}],
            "source_key": "jobs/job-2/input/source.mp4",
            "video_face_identities": "invalid",
        }
    )

    assert restored.video_face_identities is None
