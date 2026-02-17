"""Tests for scene_bundle: keyframe selection, derived features, overlay generation, and bundle builder."""

import math
from io import BytesIO

import pytest

from app.scene_bundle import (
    DerivedFrameFeatures,
    FrameData,
    SceneBundle,
    TrackMeta,
    build_scene_bundle,
    build_tracks_index,
    compute_bbox_area_pct,
    compute_bbox_centroid,
    compute_derived_features_for_frame,
    compute_iou,
    compute_polygon_area,
    compute_polygon_area_pct,
    compute_spatial_ordering,
    generate_overlay_image,
    select_keyframes,
)


# ---------------------------------------------------------------------------
# Helper: build fake scene frame dicts
# ---------------------------------------------------------------------------


def _make_frame(
    frame_id: int,
    timestamp: str = "00:00:00.000",
    detections: list | None = None,
    faces: list | None = None,
    segmentation: list | None = None,
) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "files": {
            "original": f"jobs/test/frames/original/frame_{frame_id}.jpg",
            "detection": f"jobs/test/frames/det/frame_{frame_id}.jpg",
            "segmentation": f"jobs/test/frames/seg/frame_{frame_id}.jpg",
            "face": f"jobs/test/frames/face/frame_{frame_id}.jpg",
        },
        "analysis": {
            "object_detection": detections or [],
            "face_recognition": faces or [],
            "semantic_segmentation": segmentation or [],
        },
        "analysis_artifacts": {
            "json": f"jobs/test/analysis/json/frame_{frame_id}.json"
        },
    }


def _make_detection(
    track_id: str, label: str, box: list, conf: float = 0.9, **kwargs
) -> dict:
    det = {
        "track_id": track_id,
        "label": label,
        "confidence": conf,
        "box": box,
    }
    det.update(kwargs)
    return det


def _make_face(identity_id: str, coords: list, conf: float = 0.95, **kwargs) -> dict:
    face = {
        "face_id": 0,
        "identity_id": identity_id,
        "confidence": conf,
        "coordinates": coords,
    }
    face.update(kwargs)
    return face


# ===========================================================================
# Keyframe Selection Tests
# ===========================================================================


class TestKeyframeSelection:
    def test_first_and_last_always_included(self):
        frames = [_make_frame(i, f"00:00:{i:02d}.000") for i in range(31)]
        selected = select_keyframes(frames, max_keyframes=12)
        assert 0 in selected
        assert 30 in selected

    def test_track_appearance_triggers_keyframe(self):
        """When a new track appears at frame 12, that frame should be selected."""
        frames = []
        for i in range(20):
            dets = [_make_detection("track_a", "person", [10, 10, 100, 200])]
            if i >= 12:
                dets.append(_make_detection("track_b", "car", [300, 100, 500, 300]))
            frames.append(_make_frame(i, f"00:00:{i:02d}.000", detections=dets))
        selected = select_keyframes(frames, max_keyframes=12)
        assert 12 in selected

    def test_frame_cap_enforced(self):
        """Even with many trigger events, cap must be respected."""
        frames = []
        for i in range(30):
            dets = [
                _make_detection(f"track_{i % 5}", "obj", [10 * i, 10, 10 * i + 50, 200])
            ]
            frames.append(_make_frame(i, f"00:00:{i:02d}.000", detections=dets))
        selected = select_keyframes(frames, max_keyframes=5)
        assert len(selected) <= 5

    def test_empty_scene(self):
        assert select_keyframes([]) == []

    def test_single_frame(self):
        frames = [_make_frame(0)]
        selected = select_keyframes(frames)
        assert selected == [0]

    def test_two_frames(self):
        frames = [_make_frame(0), _make_frame(1)]
        selected = select_keyframes(frames)
        assert selected == [0, 1]

    def test_interaction_proximity_triggers(self):
        """Two tracks getting close should trigger a keyframe."""
        frames = []
        for i in range(10):
            dist = 500 - i * 60  # tracks converge
            dets = [
                _make_detection("track_a", "person", [100, 100, 200, 200]),
                _make_detection(
                    "track_b", "person", [100 + dist, 100, 200 + dist, 200]
                ),
            ]
            frames.append(_make_frame(i, f"00:00:{i:02d}.000", detections=dets))
        selected = select_keyframes(
            frames, max_keyframes=12, interaction_distance_threshold=150.0
        )
        # The frame where distance drops below threshold should be selected
        assert len(selected) >= 2


# ===========================================================================
# Derived Features Tests
# ===========================================================================


class TestDerivedFeatures:
    def test_centroid_calculation(self):
        centroid = compute_bbox_centroid([90, 160, 240, 420])
        assert centroid == (165.0, 290.0)

    def test_bbox_area_pct(self):
        pct = compute_bbox_area_pct([90, 160, 240, 420], 1920, 1080)
        expected = ((240 - 90) * (420 - 160)) / (1920 * 1080) * 100.0
        assert abs(pct - expected) < 0.001

    def test_bbox_area_pct_zero_frame(self):
        assert compute_bbox_area_pct([0, 0, 100, 100], 0, 0) == 0.0

    def test_polygon_area(self):
        # Simple rectangle: 100x100 = 10000
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert compute_polygon_area(polygon) == 10000.0

    def test_polygon_area_pct(self):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        pct = compute_polygon_area_pct(polygon, 1000, 1000)
        assert abs(pct - 1.0) < 0.001  # 10000 / 1000000 * 100 = 1.0%

    def test_iou_complete_overlap(self):
        assert compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_iou_no_overlap(self):
        assert compute_iou([0, 0, 50, 50], [100, 100, 200, 200]) == 0.0

    def test_iou_partial_overlap(self):
        iou = compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0.0 < iou < 1.0

    def test_spatial_ordering(self):
        ordering = compute_spatial_ordering("A", (165, 290), "B", (500, 300))
        predicates = {o["predicate"] for o in ordering}
        assert "left_of" in predicates
        assert "above" in predicates

    def test_spatial_ordering_reversed(self):
        ordering = compute_spatial_ordering("A", (500, 300), "B", (165, 290))
        predicates = {o["predicate"] for o in ordering}
        assert "right_of" in predicates
        assert "below" in predicates

    def test_velocity_computation(self):
        frame = _make_frame(
            5,
            "00:00:05.000",
            detections=[_make_detection("track_a", "person", [140, 190, 160, 260])],
        )
        features = compute_derived_features_for_frame(
            frame,
            prev_centroids={"track_a": (100.0, 200.0)},
            prev_timestamp=0.0,
        )
        # Displacement from (100,200) to (150,225) over 5 seconds
        centroid_new = features.centroids["track_a"]
        displacement = math.sqrt(
            (centroid_new[0] - 100.0) ** 2 + (centroid_new[1] - 200.0) ** 2
        )
        expected_vel = displacement / 5.0
        assert abs(features.velocities["track_a"] - round(expected_vel, 2)) < 0.1

    def test_velocity_zero_dt(self):
        """Velocity should not be computed when dt=0."""
        frame = _make_frame(
            0,
            "00:00:00.000",
            detections=[_make_detection("track_a", "person", [100, 200, 200, 300])],
        )
        features = compute_derived_features_for_frame(
            frame,
            prev_centroids={"track_a": (150.0, 250.0)},
            prev_timestamp=0.0,
        )
        assert "track_a" not in features.velocities

    def test_pairwise_distances(self):
        frame = _make_frame(
            0,
            "00:00:00.000",
            detections=[
                _make_detection("track_a", "person", [100, 100, 200, 200]),
                _make_detection("track_b", "car", [500, 100, 600, 200]),
            ],
        )
        features = compute_derived_features_for_frame(frame)
        assert len(features.pairwise_distances) == 1
        assert features.pairwise_distances[0]["track_a"] == "track_a"
        assert features.pairwise_distances[0]["track_b"] == "track_b"
        assert features.pairwise_distances[0]["distance"] > 0

    def test_near_edges(self):
        frame = _make_frame(
            0,
            "00:00:00.000",
            detections=[
                _make_detection("track_a", "person", [100, 100, 200, 200]),
                _make_detection("track_b", "person", [120, 100, 220, 200]),
            ],
        )
        features = compute_derived_features_for_frame(frame, near_threshold=200.0)
        assert len(features.near_edges) >= 1


# ===========================================================================
# Overlay Generation Tests
# ===========================================================================


class TestOverlayGeneration:
    def _make_test_image(self, width: int = 640, height: int = 480) -> bytes:
        from PIL import Image

        img = Image.new("RGB", (width, height), color=(128, 128, 128))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_overlay_produces_image_output(self):
        img_bytes = self._make_test_image()
        frame = _make_frame(
            0,
            detections=[
                _make_detection("track_1", "person", [50, 50, 200, 300]),
                _make_detection("track_2", "car", [300, 100, 500, 250]),
            ],
        )
        result = generate_overlay_image(img_bytes, frame)
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Verify it's a valid PNG
        from PIL import Image

        img = Image.open(BytesIO(result))
        assert img.format == "PNG"
        assert img.size == (640, 480)

    def test_overlay_with_faces(self):
        img_bytes = self._make_test_image()
        frame = _make_frame(
            0,
            faces=[
                _make_face("video_person_7", [100, 50, 200, 150]),
            ],
        )
        result = generate_overlay_image(img_bytes, frame)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_overlay_with_segmentation(self):
        img_bytes = self._make_test_image()
        frame = _make_frame(
            0,
            segmentation=[
                {
                    "object_id": 1,
                    "class": "person",
                    "mask_polygon": [[50, 50], [200, 50], [200, 300], [50, 300]],
                },
            ],
        )
        result = generate_overlay_image(img_bytes, frame)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_overlay_handles_missing_segmentation(self):
        img_bytes = self._make_test_image()
        frame = _make_frame(
            0,
            detections=[_make_detection("track_1", "person", [50, 50, 200, 300])],
            segmentation=[],
        )
        result = generate_overlay_image(img_bytes, frame)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_overlay_empty_frame(self):
        img_bytes = self._make_test_image()
        frame = _make_frame(0)
        result = generate_overlay_image(img_bytes, frame)
        assert isinstance(result, bytes)
        assert len(result) > 0


# ===========================================================================
# Tracks Index Tests
# ===========================================================================


class TestTracksIndex:
    def test_build_tracks_from_detections(self):
        frames = [
            _make_frame(
                0,
                detections=[
                    _make_detection(
                        "track_a",
                        "person",
                        [10, 10, 100, 200],
                        object_track_id="obj_track_a",
                    ),
                ],
            ),
            _make_frame(
                1,
                detections=[
                    _make_detection(
                        "track_a",
                        "person",
                        [15, 15, 105, 205],
                        object_track_id="obj_track_a",
                    ),
                ],
            ),
        ]
        tracks = build_tracks_index(frames, job_id="test", scene_id="0")
        assert "obj_track_a" in tracks
        assert tracks["obj_track_a"].label == "person"
        assert len(tracks["obj_track_a"].frame_ids) == 2

    def test_build_tracks_from_faces(self):
        frames = [
            _make_frame(
                0,
                faces=[_make_face("video_person_7", [100, 50, 200, 150])],
            ),
        ]
        tracks = build_tracks_index(frames, job_id="test", scene_id="0")
        assert "video_person_7" in tracks
        assert tracks["video_person_7"].entity_type == "person"


# ===========================================================================
# SceneBundle Builder Integration Test
# ===========================================================================


class TestBuildSceneBundle:
    def test_builds_bundle_without_media_store(self):
        """Build a scene bundle without a real media store (no overlay/URL generation)."""
        frames = [
            _make_frame(
                i,
                f"00:00:{i:02d}.000",
                detections=[
                    _make_detection(
                        "track_a", "person", [10 + i * 5, 10, 100 + i * 5, 200]
                    ),
                ],
            )
            for i in range(10)
        ]
        bundle = build_scene_bundle(
            job_id="test_job",
            scene_id=0,
            source_key="input/source.mp4",
            start_sec=0.0,
            end_sec=10.0,
            scene_frames=frames,
            media_store=None,
            max_keyframes=5,
        )
        assert isinstance(bundle, SceneBundle)
        assert bundle.scene_id == "0"
        assert bundle.job_id == "test_job"
        assert len(bundle.selected_frame_ids) <= 5
        assert 0 in bundle.selected_frame_ids
        assert 9 in bundle.selected_frame_ids
        assert len(bundle.tracks_index) > 0
        assert len(bundle.derived_features) > 0
