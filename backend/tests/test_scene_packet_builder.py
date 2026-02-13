"""Tests for identity-aware entity aggregation in scene packet builder."""

from app.scene_packet_builder import collect_corpus_entities


def test_collect_corpus_entities_prefers_video_identity_metadata():
    scene_frames = [
        {
            "frame_id": 0,
            "timestamp": "00:00:01.000",
            "analysis": {
                "object_detection": [],
                "face_recognition": [
                    {
                        "face_id": 1,
                        "identity_id": "face_legacy",
                        "scene_person_id": "scene_0_person_1",
                        "video_person_id": "video_person_7",
                        "confidence": 0.93,
                        "coordinates": [1, 2, 20, 30],
                    }
                ],
            },
            "analysis_artifacts": {
                "json": "jobs/job-1/analysis/json/frame_0.json",
            },
        }
    ]

    entities, _frame_entity_ids = collect_corpus_entities(
        job_id="job-1",
        scene_id=0,
        scene_frames=scene_frames,
        start_sec=0.0,
        end_sec=2.0,
    )

    assert len(entities) == 1
    person = entities[0]
    assert person["entity_type"] == "person"
    assert person["label"] == "video_person_7"
    assert person["identity_id"] == "video_person_7"
