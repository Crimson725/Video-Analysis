"""Unit tests for scene/video face identity aggregation utilities."""

import numpy as np

from app.face_identity import (
    FaceObservation,
    aggregate_scene_identities,
    stitch_video_identities,
)


def _norm(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _obs(
    *,
    scene_id: int,
    frame_id: int,
    face_id: int,
    embedding: np.ndarray,
) -> FaceObservation:
    return FaceObservation(
        scene_id=scene_id,
        frame_id=frame_id,
        timestamp="00:00:00.000",
        face_id=face_id,
        coordinates=[0, 0, 10, 10],
        confidence=0.99,
        embedding=embedding,
        source="tracking",
    )


def test_scene_identity_aggregation_stays_stable_across_large_frame_gaps():
    observations = [
        _obs(scene_id=0, frame_id=1, face_id=1, embedding=_norm([1.0, 0.0, 0.0])),
        _obs(scene_id=0, frame_id=50, face_id=1, embedding=_norm([0.98, 0.02, 0.0])),
    ]

    assignments, clusters = aggregate_scene_identities(
        observations,
        similarity_threshold=0.75,
        ambiguity_margin=0.02,
    )

    first = assignments[(0, 1, 1)]["scene_person_id"]
    second = assignments[(0, 50, 1)]["scene_person_id"]
    assert first == second
    assert len(clusters[0]) == 1


def test_scene_identity_aggregation_flags_ambiguous_matches():
    observations = [
        _obs(scene_id=0, frame_id=1, face_id=1, embedding=_norm([1.0, 0.0, 0.0])),
        _obs(scene_id=0, frame_id=2, face_id=1, embedding=_norm([0.0, 1.0, 0.0])),
        _obs(scene_id=0, frame_id=3, face_id=1, embedding=_norm([0.72, 0.70, 0.0])),
    ]

    assignments, clusters = aggregate_scene_identities(
        observations,
        similarity_threshold=0.5,
        ambiguity_margin=0.05,
    )

    third = assignments[(0, 3, 1)]
    assert third["is_identity_ambiguous"] is True
    assert len(clusters[0]) == 3


def test_video_identity_stitching_merges_scene_clusters_for_same_person():
    observations = [
        _obs(scene_id=0, frame_id=1, face_id=1, embedding=_norm([1.0, 0.0, 0.0])),
        _obs(scene_id=1, frame_id=1, face_id=1, embedding=_norm([0.99, 0.01, 0.0])),
    ]
    _, clusters = aggregate_scene_identities(
        observations,
        similarity_threshold=0.75,
        ambiguity_margin=0.02,
    )

    scene_to_video, summary = stitch_video_identities(
        clusters,
        similarity_threshold=0.8,
        ambiguity_margin=0.01,
    )

    left = scene_to_video["scene_0_person_1"]["video_person_id"]
    right = scene_to_video["scene_1_person_1"]["video_person_id"]
    assert left == right
    assert len(summary) == 1
    assert sorted(summary[0]["scene_person_ids"]) == ["scene_0_person_1", "scene_1_person_1"]


def test_video_identity_stitching_splits_distinct_people():
    observations = [
        _obs(scene_id=0, frame_id=1, face_id=1, embedding=_norm([1.0, 0.0, 0.0])),
        _obs(scene_id=1, frame_id=1, face_id=1, embedding=_norm([0.0, 1.0, 0.0])),
    ]
    _, clusters = aggregate_scene_identities(
        observations,
        similarity_threshold=0.75,
        ambiguity_margin=0.02,
    )

    scene_to_video, summary = stitch_video_identities(
        clusters,
        similarity_threshold=0.95,
        ambiguity_margin=0.01,
    )

    left = scene_to_video["scene_0_person_1"]["video_person_id"]
    right = scene_to_video["scene_1_person_1"]["video_person_id"]
    assert left != right
    assert len(summary) == 2
