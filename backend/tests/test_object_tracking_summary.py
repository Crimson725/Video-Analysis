"""Unit tests for job-level object track aggregation."""

from copy import deepcopy

from app.analysis import run_object_tracking_summary


def _frame(frame_id: int, timestamp: str, detections: list[dict]) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "analysis": {
            "object_detection": detections,
        },
    }


def test_object_tracking_summary_groups_multi_label_tracks_and_enriches_detections():
    frames = [
        _frame(
            0,
            "00:00:00.500",
            [
                {
                    "track_id": "car_7",
                    "label": "car",
                    "confidence": 0.9,
                    "box": [10, 20, 30, 40],
                }
            ],
        ),
        _frame(
            1,
            "00:00:01.000",
            [
                {
                    "track_id": "car_7",
                    "label": "car",
                    "confidence": 0.88,
                    "box": [12, 22, 32, 42],
                },
                {
                    "track_id": "dog_3",
                    "label": "dog",
                    "confidence": 0.86,
                    "box": [40, 20, 70, 60],
                },
            ],
        ),
    ]

    summary = run_object_tracking_summary(frame_results=frames, job_id="job-1")

    assert summary["enabled"] is True
    assert summary["method"] == "object_tracking_v1"
    assert len(summary["tracks"]) == 2
    assert {track["label"] for track in summary["tracks"]} == {"car", "dog"}

    car_track = next(track for track in summary["tracks"] if track["label"] == "car")
    assert car_track["frame_span"]["first_frame_id"] == 0
    assert car_track["frame_span"]["last_frame_id"] == 1
    assert car_track["confidence"]["samples"] == 2
    assert len(car_track["evidence"]) == 2

    frame_0_car = frames[0]["analysis"]["object_detection"][0]
    frame_1_car = frames[1]["analysis"]["object_detection"][0]
    assert frame_0_car["object_track_id"] == car_track["object_track_id"]
    assert frame_1_car["object_track_id"] == car_track["object_track_id"]


def test_object_tracking_summary_returns_empty_tracks_when_no_detections():
    frames = [
        _frame(0, "00:00:00.500", []),
        _frame(1, "00:00:01.000", []),
    ]

    summary = run_object_tracking_summary(frame_results=frames, job_id="job-2")

    assert summary == {
        "enabled": True,
        "method": "object_tracking_v1",
        "tracks": [],
    }


def test_object_tracking_summary_is_deterministic_for_identical_inputs():
    frames = [
        _frame(
            0,
            "00:00:00.500",
            [
                {
                    "track_id": "car_7",
                    "label": "car",
                    "confidence": 0.9,
                    "box": [10, 20, 30, 40],
                },
                {
                    "track_id": "person_1",
                    "label": "person",
                    "confidence": 0.91,
                    "box": [1, 2, 20, 30],
                },
            ],
        ),
    ]

    first = run_object_tracking_summary(frame_results=deepcopy(frames), job_id="job-3")
    second = run_object_tracking_summary(frame_results=deepcopy(frames), job_id="job-3")

    assert first == second


def test_object_tracking_summary_keeps_same_canonical_id_across_chunk_boundary():
    frames = [
        _frame(
            299,
            "00:04:59.000",
            [
                {
                    "track_id": "car_chunk0_3",
                    "label": "car",
                    "confidence": 0.92,
                    "box": [100, 120, 220, 260],
                }
            ],
        ),
        _frame(
            300,
            "00:05:00.000",
            [
                {
                    "track_id": "car_chunk1_1",
                    "label": "car",
                    "confidence": 0.90,
                    "box": [104, 122, 224, 262],
                }
            ],
        ),
    ]

    summary = run_object_tracking_summary(frame_results=frames, job_id="job-boundary")

    assert len(summary["tracks"]) == 1
    track = summary["tracks"][0]
    assert track["source_track_ids"] == ["car_chunk0_3", "car_chunk1_1"]
    assert track["is_identity_ambiguous"] is False
    assert frames[0]["analysis"]["object_detection"][0]["object_track_id"] == track[
        "object_track_id"
    ]
    assert frames[1]["analysis"]["object_detection"][0]["object_track_id"] == track[
        "object_track_id"
    ]


def test_object_tracking_summary_keeps_same_id_after_short_occlusion_gap():
    frames = [
        _frame(
            10,
            "00:00:10.000",
            [
                {
                    "track_id": "person_a",
                    "label": "person",
                    "confidence": 0.89,
                    "box": [40, 30, 120, 220],
                }
            ],
        ),
        _frame(11, "00:00:11.000", []),
        _frame(
            12,
            "00:00:12.000",
            [
                {
                    "track_id": "person_b",
                    "label": "person",
                    "confidence": 0.91,
                    "box": [43, 32, 123, 222],
                }
            ],
        ),
    ]

    summary = run_object_tracking_summary(frame_results=frames, job_id="job-occlusion")

    assert len(summary["tracks"]) == 1
    track = summary["tracks"][0]
    assert track["source_track_ids"] == ["person_a", "person_b"]
    assert track["identity_confidence"] is not None
    assert track["identity_confidence"] >= 0.8
