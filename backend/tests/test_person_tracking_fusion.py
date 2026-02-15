"""Unit tests for object+face fused person tracking."""

from app.analysis import run_person_tracking_fusion


def _frame(
    frame_id: int,
    *,
    track_id: str,
    box: list[int],
    faces: list[dict],
    timestamp: str,
) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "analysis": {
            "semantic_segmentation": [],
            "object_detection": [
                {
                    "track_id": track_id,
                    "label": "person",
                    "confidence": 0.9,
                    "box": box,
                }
            ],
            "face_recognition": faces,
            "enrichment": {},
        },
    }


def test_fusion_merges_object_tracks_when_video_identity_matches():
    frames = [
        _frame(
            0,
            track_id="person_1",
            box=[10, 10, 110, 210],
            faces=[
                {
                    "face_id": 1,
                    "identity_id": "face_1",
                    "video_person_id": "video_person_7",
                    "match_confidence": 0.95,
                    "coordinates": [30, 30, 70, 90],
                }
            ],
            timestamp="00:00:00.500",
        ),
        _frame(
            1,
            track_id="person_9",
            box=[15, 12, 115, 212],
            faces=[
                {
                    "face_id": 1,
                    "identity_id": "face_9",
                    "video_person_id": "video_person_7",
                    "match_confidence": 0.92,
                    "coordinates": [35, 30, 75, 88],
                }
            ],
            timestamp="00:00:01.000",
        ),
    ]

    fused = run_person_tracking_fusion(frame_results=frames, job_id="job-1")

    assert fused["enabled"] is True
    assert fused["method"] == "object_face_fusion_v1"
    assert len(fused["tracks"]) == 1
    track = fused["tracks"][0]
    assert track["identity_id"] == "video_person_7"
    assert track["object_track_ids"] == ["person_1", "person_9"]

    first_detection = frames[0]["analysis"]["object_detection"][0]
    second_detection = frames[1]["analysis"]["object_detection"][0]
    assert first_detection["person_track_id"] == second_detection["person_track_id"]
    assert first_detection["person_identity_id"] == "video_person_7"


def test_fusion_keeps_identity_unresolved_without_face_matches():
    frames = [
        _frame(
            0,
            track_id="person_3",
            box=[20, 20, 120, 220],
            faces=[],
            timestamp="00:00:00.200",
        )
    ]

    fused = run_person_tracking_fusion(frame_results=frames, job_id="job-2")

    assert len(fused["tracks"]) == 1
    track = fused["tracks"][0]
    assert track["identity_id"] is None
    assert track["person_track_id"].startswith("person_track_")

    detection = frames[0]["analysis"]["object_detection"][0]
    assert detection["person_track_id"] == track["person_track_id"]
    assert detection["person_identity_id"] is None


def test_fusion_marks_track_ambiguous_for_conflicting_identity_votes():
    frames = [
        _frame(
            0,
            track_id="person_11",
            box=[0, 0, 100, 200],
            faces=[
                {
                    "face_id": 1,
                    "video_person_id": "video_person_A",
                    "identity_id": "video_person_A",
                    "match_confidence": 0.90,
                    "coordinates": [20, 20, 60, 80],
                }
            ],
            timestamp="00:00:00.000",
        ),
        _frame(
            1,
            track_id="person_11",
            box=[0, 0, 100, 200],
            faces=[
                {
                    "face_id": 1,
                    "video_person_id": "video_person_B",
                    "identity_id": "video_person_B",
                    "match_confidence": 0.89,
                    "coordinates": [22, 21, 62, 81],
                }
            ],
            timestamp="00:00:00.500",
        ),
    ]

    fused = run_person_tracking_fusion(frame_results=frames, job_id="job-3")

    assert len(fused["tracks"]) == 1
    track = fused["tracks"][0]
    assert track["identity_id"] is None
    assert track["is_identity_ambiguous"] is True
