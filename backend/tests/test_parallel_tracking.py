"""Unit tests for parallel chunked tracking helpers and payload shaping."""

from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import app.parallel_tracking as parallel_tracking


def _row(
    *,
    t_ms: int,
    global_id: int,
    class_id: int = 0,
    conf: float = 0.9,
    x1: int = 0,
    y1: int = 0,
    x2: int = 20,
    y2: int = 20,
) -> parallel_tracking.CanonicalTrackRow:
    return parallel_tracking.CanonicalTrackRow(
        t_ms=t_ms,
        global_id=global_id,
        class_id=class_id,
        conf=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _summary_config(**overrides) -> parallel_tracking.ChunkedTrackingConfig:
    defaults = dict(
        enabled=True,
        chunk_duration_sec=300.0,
        overlap_sec=15.0,
        sample_fps=10,
        chunk_max_workers=2,
        detector_weights="yolo11n.pt",
        tracker_config="botsort_reid.yaml",
        backend_strategy="default",
        stitch_strategy="default",
        zone_strategy="grid3x3",
        confidence_threshold=0.05,
        min_cosine=0.3,
        min_iou=0.1,
        velocity_window=3,
        use_clip_embeddings=False,
        clip_model_id="ViT-B-32",
        clip_pretrained="openai",
        ground_truth_backend="sqlite",
        output_mode="summary_v2",
        top_entities_per_scene=30,
        path_grid_width=12,
        path_grid_height=7,
        path_max_segments=6,
    )
    defaults.update(overrides)
    return parallel_tracking.ChunkedTrackingConfig(**defaults)


def test_plan_chunks_uses_fixed_duration_with_overlap():
    chunks = parallel_tracking._plan_chunks(
        duration_sec=620.0,
        chunk_duration_sec=300.0,
        overlap_sec=15.0,
    )

    assert [(chunk.start_sec, chunk.end_sec) for chunk in chunks] == [
        (0.0, 300.0),
        (285.0, 585.0),
        (570.0, 620.0),
    ]


def test_stitch_boundary_matches_track_when_thresholds_are_met():
    active = {
        "global_1": parallel_tracking._GlobalTrackState(
            global_id="global_1",
            class_id=2,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            bbox_end=[10, 10, 30, 30],
            velocity_xy=(0.0, 0.0),
            last_t=100.0,
        )
    }
    local = {
        "2:9": parallel_tracking._TrackSignature(
            local_id="2:9",
            class_id=2,
            start_t=101.0,
            end_t=102.0,
            bbox_start=[10, 10, 30, 30],
            bbox_end=[10, 10, 30, 30],
            velocity_xy=(0.0, 0.0),
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
    }

    mapping, next_index = parallel_tracking._stitch_boundary(
        active_global_tracks=active,
        local_head_signatures=local,
        min_cosine=0.3,
        min_iou=0.1,
        overlap_sec=15.0,
        next_global_index=2,
    )

    assert mapping["2:9"] == "global_1"
    assert next_index == 2


def test_stitch_boundary_creates_new_global_id_on_failed_match():
    active = {
        "global_1": parallel_tracking._GlobalTrackState(
            global_id="global_1",
            class_id=2,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            bbox_end=[0, 0, 20, 20],
            velocity_xy=(0.0, 0.0),
            last_t=100.0,
        )
    }
    local = {
        "2:9": parallel_tracking._TrackSignature(
            local_id="2:9",
            class_id=2,
            start_t=101.0,
            end_t=102.0,
            bbox_start=[200, 200, 240, 240],
            bbox_end=[200, 200, 240, 240],
            velocity_xy=(0.0, 0.0),
            embedding=np.array([-1.0, 0.0], dtype=np.float32),
        )
    }

    mapping, next_index = parallel_tracking._stitch_boundary(
        active_global_tracks=active,
        local_head_signatures=local,
        min_cosine=0.3,
        min_iou=0.1,
        overlap_sec=15.0,
        next_global_index=2,
    )

    assert mapping["2:9"] == "global_2"
    assert next_index == 3


def test_unknown_tracking_strategy_fails_fast_with_clear_error():
    settings = SimpleNamespace(
        enable_parallel_chunked_tracking_pipeline=True,
        parallel_tracking_backend_strategy="unknown-backend",
    )

    with pytest.raises(ValueError, match="Invalid tracking_backend strategy"):
        parallel_tracking.run_parallel_chunked_tracking(
            video_path="/tmp/video.mp4",
            settings=settings,
            scenes=[],
            output_dir="/tmp",
        )


def test_run_parallel_chunked_tracking_emits_video_summary_and_persists_ground_truth(
    monkeypatch, tmp_path
):
    settings = SimpleNamespace(
        enable_parallel_chunked_tracking_pipeline=True,
        parallel_tracking_use_clip_embeddings=False,
        parallel_tracking_output_mode="summary_v2",
        parallel_tracking_chunk_duration_sec=300,
        parallel_tracking_overlap_sec=15,
        parallel_tracking_sample_fps=10,
        parallel_tracking_chunk_max_workers=2,
        parallel_tracking_min_cosine=0.0,
        parallel_tracking_min_iou=0.1,
    )

    class _DummyCap:
        def __init__(self, _video_path: str) -> None:
            pass

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 6000.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1000.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 1000.0
            return 0.0

        def release(self) -> None:
            return None

    monkeypatch.setattr(parallel_tracking.cv2, "VideoCapture", _DummyCap)
    monkeypatch.setattr(
        parallel_tracking,
        "_build_embedder",
        lambda _config: (parallel_tracking._FallbackAppearanceEmbedder(), "fallback"),
    )
    monkeypatch.setattr(
        parallel_tracking,
        "_plan_chunks",
        lambda *_args, **_kwargs: [
            parallel_tracking._ChunkWindow(chunk_id=0, start_sec=0.0, end_sec=300.0),
            parallel_tracking._ChunkWindow(chunk_id=1, start_sec=285.0, end_sec=585.0),
        ],
    )

    track_a = {
        "class_id": 1,
        "observations": [
            parallel_tracking._TrackObservation(
                t_sec=289.0,
                frame_idx=2890,
                bbox_xyxy=[10, 10, 40, 40],
                confidence=0.5,
                embedding=None,
            ),
            parallel_tracking._TrackObservation(
                t_sec=289.1,
                frame_idx=2891,
                bbox_xyxy=[10, 10, 40, 40],
                confidence=0.5,
                embedding=None,
            ),
            parallel_tracking._TrackObservation(
                t_sec=289.2,
                frame_idx=2892,
                bbox_xyxy=[10, 10, 40, 40],
                confidence=0.5,
                embedding=None,
            )
        ],
    }
    track_b = {
        "class_id": 1,
        "observations": [
            parallel_tracking._TrackObservation(
                t_sec=289.0,
                frame_idx=2890,
                bbox_xyxy=[10, 10, 40, 40],
                confidence=0.9,
                embedding=None,
            ),
            parallel_tracking._TrackObservation(
                t_sec=289.1,
                frame_idx=2891,
                bbox_xyxy=[10, 10, 40, 40],
                confidence=0.9,
                embedding=None,
            ),
            parallel_tracking._TrackObservation(
                t_sec=289.2,
                frame_idx=2892,
                bbox_xyxy=[10, 10, 40, 40],
                confidence=0.9,
                embedding=None,
            )
        ],
    }

    def _fake_extract_chunk_local_tracks(
        *,
        video_path: str,
        chunk: parallel_tracking._ChunkWindow,
        config: parallel_tracking.ChunkedTrackingConfig,
        native_fps: float,
        embedder: object,
    ) -> tuple[list[dict], dict[str, dict], dict[int, str]]:
        del video_path, config, native_fps, embedder
        if chunk.chunk_id == 0:
            return (
                [
                    {
                        "t_sec": 289.0,
                        "frame_idx": 2890,
                        "local_id": "1:a",
                        "class_id": 1,
                        "conf": 0.5,
                        "bbox_xyxy": [10, 10, 40, 40],
                    },
                    {
                        "t_sec": 289.1,
                        "frame_idx": 2891,
                        "local_id": "1:a",
                        "class_id": 1,
                        "conf": 0.5,
                        "bbox_xyxy": [10, 10, 40, 40],
                    },
                    {
                        "t_sec": 289.2,
                        "frame_idx": 2892,
                        "local_id": "1:a",
                        "class_id": 1,
                        "conf": 0.5,
                        "bbox_xyxy": [10, 10, 40, 40],
                    }
                ],
                {"1:a": track_a},
                {1: "person"},
            )
        return (
            [
                {
                    "t_sec": 289.0,
                    "frame_idx": 2890,
                    "local_id": "1:b",
                    "class_id": 1,
                    "conf": 0.9,
                    "bbox_xyxy": [10, 10, 40, 40],
                },
                {
                    "t_sec": 289.1,
                    "frame_idx": 2891,
                    "local_id": "1:b",
                    "class_id": 1,
                    "conf": 0.9,
                    "bbox_xyxy": [10, 10, 40, 40],
                },
                {
                    "t_sec": 289.2,
                    "frame_idx": 2892,
                    "local_id": "1:b",
                    "class_id": 1,
                    "conf": 0.9,
                    "bbox_xyxy": [10, 10, 40, 40],
                }
            ],
            {"1:b": track_b},
            {1: "person"},
        )

    monkeypatch.setattr(
        parallel_tracking,
        "_extract_chunk_local_tracks",
        _fake_extract_chunk_local_tracks,
    )

    payload = parallel_tracking.run_parallel_chunked_tracking(
        video_path="/tmp/video.mp4",
        settings=settings,
        scenes=[(288.5, 290.5)],
        output_dir=str(tmp_path),
    )

    assert payload["enabled"] is True
    assert payload["output_mode"] == "summary_v2"
    assert payload["stats"]["row_count"] == 3
    assert payload["stats"]["track_count"] == 0
    assert payload["tracks"] == []
    assert payload["stats"]["scene_count"] == 0
    assert payload["zone_definition"]["layout"] == "3x3"
    assert payload["zone_definition"]["labels"][0] == "top-left"
    assert len(payload["zone_definition"]["zones"]) == 9
    assert len(payload["entities"]) == 1
    entity = payload["entities"][0]
    assert entity["entity_id"] == "person-1"
    assert entity["label"] == "person"
    assert entity["appearance_ranges_ms"] == [{"start_ms": 289000, "end_ms": 289200}]
    assert entity["zones_visited"] == ["top-left"]
    assert entity["zone_occupancy"]["top-left"] == 3
    assert entity["zone_transitions"] == []
    assert payload["artifacts"]["canonical_sqlite"].endswith("tracks.canonical.sqlite3")
    assert payload["artifacts"]["video_summary_json"].endswith(
        "tracks.video_summary.json"
    )
    assert Path(payload["artifacts"]["canonical_sqlite"]).is_file()
    assert Path(payload["artifacts"]["video_summary_json"]).is_file()


def test_run_parallel_chunked_tracking_legacy_mode_preserves_compact_payload(
    monkeypatch, tmp_path
):
    settings = SimpleNamespace(
        enable_parallel_chunked_tracking_pipeline=True,
        parallel_tracking_use_clip_embeddings=False,
        parallel_tracking_output_mode="legacy",
        parallel_tracking_chunk_duration_sec=300,
        parallel_tracking_overlap_sec=15,
        parallel_tracking_sample_fps=10,
        parallel_tracking_min_cosine=0.0,
        parallel_tracking_min_iou=0.1,
    )

    class _DummyCap:
        def __init__(self, _video_path: str) -> None:
            pass

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 1200.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1000.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 1000.0
            return 0.0

        def release(self) -> None:
            return None

    monkeypatch.setattr(parallel_tracking.cv2, "VideoCapture", _DummyCap)
    monkeypatch.setattr(
        parallel_tracking,
        "_build_embedder",
        lambda _config: (parallel_tracking._FallbackAppearanceEmbedder(), "fallback"),
    )
    monkeypatch.setattr(
        parallel_tracking,
        "_plan_chunks",
        lambda *_args, **_kwargs: [
            parallel_tracking._ChunkWindow(chunk_id=0, start_sec=0.0, end_sec=30.0),
        ],
    )

    def _fake_extract_chunk_local_tracks(**_kwargs):
        track = {
            "class_id": 1,
            "observations": [
                parallel_tracking._TrackObservation(
                    t_sec=1.0,
                    frame_idx=10,
                    bbox_xyxy=[100, 100, 220, 220],
                    confidence=0.8,
                    embedding=None,
                ),
                parallel_tracking._TrackObservation(
                    t_sec=2.0,
                    frame_idx=20,
                    bbox_xyxy=[100, 100, 220, 220],
                    confidence=0.8,
                    embedding=None,
                ),
                parallel_tracking._TrackObservation(
                    t_sec=3.0,
                    frame_idx=30,
                    bbox_xyxy=[100, 100, 220, 220],
                    confidence=0.8,
                    embedding=None,
                ),
            ],
        }
        return (
            [
                {
                    "t_sec": 1.0,
                    "frame_idx": 10,
                    "local_id": "1:a",
                    "class_id": 1,
                    "conf": 0.8,
                    "bbox_xyxy": [100, 100, 220, 220],
                },
                {
                    "t_sec": 2.0,
                    "frame_idx": 20,
                    "local_id": "1:a",
                    "class_id": 1,
                    "conf": 0.8,
                    "bbox_xyxy": [100, 100, 220, 220],
                },
                {
                    "t_sec": 3.0,
                    "frame_idx": 30,
                    "local_id": "1:a",
                    "class_id": 1,
                    "conf": 0.8,
                    "bbox_xyxy": [100, 100, 220, 220],
                },
            ],
            {"1:a": track},
            {1: "person"},
        )

    monkeypatch.setattr(
        parallel_tracking,
        "_extract_chunk_local_tracks",
        _fake_extract_chunk_local_tracks,
    )

    payload = parallel_tracking.run_parallel_chunked_tracking(
        video_path="/tmp/video.mp4",
        settings=settings,
        scenes=[(0.0, 10.0)],
        output_dir=str(tmp_path),
    )

    assert payload["output_mode"] == "legacy"
    assert payload["stats"]["track_count"] == 1
    assert payload["tracks"][0]["id"] == 1
    assert payload["scenes"][0]["track_ids"] == [1]
    assert payload["artifacts"]["tracks_compact_json"].endswith("tracks.compact.json")
    assert "video_summary_json" not in payload["artifacts"]


def test_run_parallel_chunked_tracking_dual_mode_writes_compare_artifacts(
    monkeypatch, tmp_path
):
    settings = SimpleNamespace(
        enable_parallel_chunked_tracking_pipeline=True,
        parallel_tracking_use_clip_embeddings=False,
        parallel_tracking_output_mode="dual",
        parallel_tracking_chunk_duration_sec=300,
        parallel_tracking_overlap_sec=15,
        parallel_tracking_sample_fps=10,
        parallel_tracking_min_cosine=0.0,
        parallel_tracking_min_iou=0.1,
    )

    class _DummyCap:
        def __init__(self, _video_path: str) -> None:
            pass

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 600.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1000.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 1000.0
            return 0.0

        def release(self) -> None:
            return None

    monkeypatch.setattr(parallel_tracking.cv2, "VideoCapture", _DummyCap)
    monkeypatch.setattr(
        parallel_tracking,
        "_build_embedder",
        lambda _config: (parallel_tracking._FallbackAppearanceEmbedder(), "fallback"),
    )
    monkeypatch.setattr(
        parallel_tracking,
        "_plan_chunks",
        lambda *_args, **_kwargs: [
            parallel_tracking._ChunkWindow(chunk_id=0, start_sec=0.0, end_sec=30.0),
        ],
    )
    monkeypatch.setattr(
        parallel_tracking,
        "_extract_chunk_local_tracks",
        lambda **_kwargs: (
            [
                {
                    "t_sec": 1.0,
                    "frame_idx": 10,
                    "local_id": "1:a",
                    "class_id": 1,
                    "conf": 0.8,
                    "bbox_xyxy": [100, 100, 220, 220],
                },
                {
                    "t_sec": 2.0,
                    "frame_idx": 20,
                    "local_id": "1:a",
                    "class_id": 1,
                    "conf": 0.8,
                    "bbox_xyxy": [140, 100, 260, 220],
                },
                {
                    "t_sec": 2.1,
                    "frame_idx": 21,
                    "local_id": "1:a",
                    "class_id": 1,
                    "conf": 0.8,
                    "bbox_xyxy": [145, 100, 265, 220],
                },
            ],
            {
                "1:a": {
                    "class_id": 1,
                    "observations": [
                        parallel_tracking._TrackObservation(
                            t_sec=1.0,
                            frame_idx=10,
                            bbox_xyxy=[100, 100, 220, 220],
                            confidence=0.8,
                            embedding=None,
                        ),
                        parallel_tracking._TrackObservation(
                            t_sec=2.0,
                            frame_idx=20,
                            bbox_xyxy=[140, 100, 260, 220],
                            confidence=0.8,
                            embedding=None,
                        ),
                        parallel_tracking._TrackObservation(
                            t_sec=2.1,
                            frame_idx=21,
                            bbox_xyxy=[145, 100, 265, 220],
                            confidence=0.8,
                            embedding=None,
                        ),
                    ],
                }
            },
            {1: "person"},
        ),
    )

    payload = parallel_tracking.run_parallel_chunked_tracking(
        video_path="/tmp/video.mp4",
        settings=settings,
        scenes=[(0.0, 10.0)],
        output_dir=str(tmp_path),
    )

    assert payload["output_mode"] == "dual"
    assert payload["tracks"] == []
    assert len(payload["entities"]) == 1
    assert payload["stats"]["legacy_track_count"] == 1
    assert payload["rollout"]["mode"] == "dual"
    assert payload["artifacts"]["tracks_compact_json"].endswith("tracks.compact.json")
    assert payload["artifacts"]["video_summary_json"].endswith(
        "tracks.video_summary.json"
    )


def test_key_box_selection_keeps_material_changes_and_max_gap():
    span_rows = [
        _row(t_ms=0, global_id=1, x1=100, y1=100, x2=300, y2=300),
        _row(t_ms=200, global_id=1, x1=102, y1=100, x2=302, y2=300),
        _row(t_ms=400, global_id=1, x1=140, y1=100, x2=340, y2=300),
        _row(t_ms=1500, global_id=1, x1=141, y1=100, x2=341, y2=300),
        _row(t_ms=1900, global_id=1, x1=142, y1=100, x2=342, y2=300),
    ]

    key_boxes = parallel_tracking._select_key_boxes(
        span_rows=span_rows,
        frame_width=1000,
        frame_height=1000,
    )

    assert [box[0] for box in key_boxes] == [0, 400, 1500, 1900]
    for box in key_boxes:
        assert len(box) == 5
        assert all(0 <= value <= 10000 for value in box[1:])


def test_build_compact_tracks_splits_spans_and_filters_noise():
    canonical_rows = [
        # Tiny flicker (filtered by area ratio).
        _row(t_ms=0, global_id=1, class_id=2, conf=0.8, x1=10, y1=10, x2=11, y2=11),
        _row(t_ms=100, global_id=1, class_id=2, conf=0.8, x1=10, y1=10, x2=11, y2=11),
        _row(t_ms=200, global_id=1, class_id=2, conf=0.8, x1=10, y1=10, x2=11, y2=11),
        # Short and sparse (filtered by duration/key-count rule).
        _row(
            t_ms=0,
            global_id=2,
            class_id=2,
            conf=0.8,
            x1=100,
            y1=100,
            x2=200,
            y2=200,
        ),
        _row(
            t_ms=200,
            global_id=2,
            class_id=2,
            conf=0.8,
            x1=100,
            y1=100,
            x2=200,
            y2=200,
        ),
        # Valid track with two visibility spans.
        _row(
            t_ms=0,
            global_id=3,
            class_id=5,
            conf=0.9,
            x1=300,
            y1=300,
            x2=500,
            y2=500,
        ),
        _row(
            t_ms=100,
            global_id=3,
            class_id=5,
            conf=0.9,
            x1=300,
            y1=300,
            x2=500,
            y2=500,
        ),
        _row(
            t_ms=200,
            global_id=3,
            class_id=5,
            conf=0.9,
            x1=300,
            y1=300,
            x2=500,
            y2=500,
        ),
        _row(
            t_ms=1000,
            global_id=3,
            class_id=5,
            conf=0.9,
            x1=300,
            y1=300,
            x2=500,
            y2=500,
        ),
        _row(
            t_ms=1100,
            global_id=3,
            class_id=5,
            conf=0.9,
            x1=300,
            y1=300,
            x2=500,
            y2=500,
        ),
        _row(
            t_ms=1200,
            global_id=3,
            class_id=5,
            conf=0.9,
            x1=300,
            y1=300,
            x2=500,
            y2=500,
        ),
    ]

    tracks = parallel_tracking._build_compact_tracks(
        canonical_rows=canonical_rows,
        sample_fps=10,
        frame_width=1000,
        frame_height=1000,
    )

    assert [track["id"] for track in tracks] == [3]
    assert len(tracks[0]["spans"]) == 2


def test_motion_classification_thresholds_are_deterministic():
    stationary_rows = [
        _row(t_ms=0, global_id=1, x1=100, y1=100, x2=200, y2=200),
        _row(t_ms=1000, global_id=1, x1=100, y1=100, x2=200, y2=200),
    ]
    moving_rows = [
        _row(t_ms=0, global_id=1, x1=100, y1=100, x2=200, y2=200),
        _row(t_ms=1000, global_id=1, x1=110, y1=100, x2=210, y2=200),
    ]
    fast_rows = [
        _row(t_ms=0, global_id=1, x1=100, y1=100, x2=200, y2=200),
        _row(t_ms=1000, global_id=1, x1=140, y1=100, x2=240, y2=200),
    ]

    assert (
        parallel_tracking._classify_motion(rows=stationary_rows, frame_diagonal=1000.0)
        == "S"
    )
    assert (
        parallel_tracking._classify_motion(rows=moving_rows, frame_diagonal=1000.0)
        == "M"
    )
    assert parallel_tracking._classify_motion(rows=fast_rows, frame_diagonal=1000.0) == "F"


def test_entity_path_encoding_merges_to_segment_cap():
    rows = [
        _row(t_ms=i * 1000, global_id=1, x1=(i * 100), y1=0, x2=(i * 100) + 40, y2=40)
        for i in range(9)
    ]

    path = parallel_tracking._encode_entity_path(
        rows=rows,
        frame_width=1200,
        frame_height=700,
        grid_width=12,
        grid_height=7,
        max_segments=6,
    )

    assert path is not None
    assert len(path.split(",")) <= 6


def test_scene_summary_applies_top_n_and_tail_counts():
    rows = [
        _row(t_ms=1000, global_id=i, class_id=0 if i <= 30 else 1, x2=20, y2=20)
        for i in range(1, 36)
    ]
    summaries = parallel_tracking._scene_summary_from_rows(
        canonical_rows=rows,
        scenes=[(1.0, 2.0)],
        class_name_map={0: "person", 1: "chair"},
        frame_width=1920,
        frame_height=1080,
        config=_summary_config(),
    )

    assert len(summaries) == 1
    scene = summaries[0]
    assert len(scene["entities_top"]) == 30
    assert [entity["id"] for entity in scene["entities_top"]] == list(range(1, 31))
    assert scene["counts_by_label_tail"] == {"chair": 5}


def test_scene_summary_is_repeatable_for_identical_input():
    rows = [
        _row(t_ms=1000, global_id=1, class_id=0, x1=100, y1=100, x2=180, y2=180),
        _row(t_ms=2000, global_id=1, class_id=0, x1=140, y1=100, x2=220, y2=180),
        _row(t_ms=3000, global_id=2, class_id=1, x1=700, y1=400, x2=760, y2=460),
    ]
    config = _summary_config()
    first = parallel_tracking._scene_summary_from_rows(
        canonical_rows=rows,
        scenes=[(1.0, 3.5)],
        class_name_map={0: "person", 1: "chair"},
        frame_width=1920,
        frame_height=1080,
        config=config,
    )
    second = parallel_tracking._scene_summary_from_rows(
        canonical_rows=rows,
        scenes=[(1.0, 3.5)],
        class_name_map={0: "person", 1: "chair"},
        frame_width=1920,
        frame_height=1080,
        config=config,
    )

    assert first == second


def test_simplified_video_summary_builds_zone_definition_and_transitions():
    rows = [
        _row(t_ms=1000, global_id=1, class_id=0, x1=10, y1=10, x2=70, y2=70),
        _row(t_ms=1100, global_id=1, class_id=0, x1=350, y1=350, x2=420, y2=420),
        _row(t_ms=1200, global_id=1, class_id=0, x1=730, y1=730, x2=790, y2=790),
    ]
    zone_definition, entities = parallel_tracking._build_simplified_video_summary(
        canonical_rows=rows,
        class_name_map={0: "person"},
        frame_width=900,
        frame_height=900,
        sample_fps=10,
        zone_definition_strategy=parallel_tracking._build_zone_definition_3x3,
    )

    assert zone_definition["layout"] == "3x3"
    assert zone_definition["labels"] == [
        "top-left",
        "top-center",
        "top-right",
        "middle-left",
        "center",
        "middle-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ]
    assert zone_definition["zones"]["center"] == {"x1": 300, "y1": 300, "x2": 600, "y2": 600}

    assert len(entities) == 1
    entity = entities[0]
    assert entity["entity_id"] == "person-1"
    assert entity["appearance_ranges_ms"] == [{"start_ms": 1000, "end_ms": 1200}]
    assert entity["zones_visited"] == ["top-left", "center", "bottom-right"]
    assert entity["zone_transitions"] == [
        {"from": "top-left", "to": "center", "at_ms": 1100},
        {"from": "center", "to": "bottom-right", "at_ms": 1200},
    ]


def test_parallel_chunk_reducer_orders_results_by_chunk_id_even_when_workers_finish_out_of_order(
    monkeypatch, tmp_path
):
    settings = SimpleNamespace(
        enable_parallel_chunked_tracking_pipeline=True,
        parallel_tracking_use_clip_embeddings=False,
        parallel_tracking_output_mode="summary_v2",
        parallel_tracking_chunk_duration_sec=300,
        parallel_tracking_overlap_sec=15,
        parallel_tracking_sample_fps=10,
        parallel_tracking_chunk_max_workers=2,
        parallel_tracking_min_cosine=0.0,
        parallel_tracking_min_iou=0.0,
    )

    class _DummyCap:
        def __init__(self, _video_path: str) -> None:
            pass

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 6000.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 900.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 900.0
            return 0.0

        def release(self) -> None:
            return None

    monkeypatch.setattr(parallel_tracking.cv2, "VideoCapture", _DummyCap)
    monkeypatch.setattr(
        parallel_tracking,
        "_build_embedder",
        lambda _config: (parallel_tracking._FallbackAppearanceEmbedder(), "fallback"),
    )
    monkeypatch.setattr(
        parallel_tracking,
        "_plan_chunks",
        lambda *_args, **_kwargs: [
            parallel_tracking._ChunkWindow(chunk_id=0, start_sec=0.0, end_sec=300.0),
            parallel_tracking._ChunkWindow(chunk_id=1, start_sec=285.0, end_sec=585.0),
        ],
    )

    completion_order: list[int] = []

    def _fake_extract_chunk_local_tracks(
        *,
        video_path: str,
        chunk: parallel_tracking._ChunkWindow,
        config: parallel_tracking.ChunkedTrackingConfig,
        native_fps: float,
        embedder: object,
    ) -> tuple[list[dict], dict[str, dict], dict[int, str]]:
        del video_path, config, native_fps, embedder
        if chunk.chunk_id == 0:
            time.sleep(0.03)
            completion_order.append(0)
            return (
                [
                    {
                        "t_sec": 289.0,
                        "frame_idx": 2890,
                        "local_id": "0:a",
                        "class_id": 0,
                        "conf": 0.9,
                        "bbox_xyxy": [20, 20, 80, 80],
                    },
                    {
                        "t_sec": 289.1,
                        "frame_idx": 2891,
                        "local_id": "0:a",
                        "class_id": 0,
                        "conf": 0.9,
                        "bbox_xyxy": [25, 25, 85, 85],
                    },
                ],
                {
                    "0:a": {
                        "class_id": 0,
                        "observations": [
                            parallel_tracking._TrackObservation(
                                t_sec=289.0,
                                frame_idx=2890,
                                bbox_xyxy=[20, 20, 80, 80],
                                confidence=0.9,
                                embedding=None,
                            ),
                            parallel_tracking._TrackObservation(
                                t_sec=289.1,
                                frame_idx=2891,
                                bbox_xyxy=[25, 25, 85, 85],
                                confidence=0.9,
                                embedding=None,
                            ),
                        ],
                    }
                },
                {0: "person"},
            )

        time.sleep(0.01)
        completion_order.append(1)
        return (
            [
                {
                    "t_sec": 289.1,
                    "frame_idx": 2891,
                    "local_id": "0:b",
                    "class_id": 0,
                    "conf": 0.95,
                    "bbox_xyxy": [25, 25, 85, 85],
                },
                {
                    "t_sec": 289.2,
                    "frame_idx": 2892,
                    "local_id": "0:b",
                    "class_id": 0,
                    "conf": 0.95,
                    "bbox_xyxy": [30, 30, 90, 90],
                },
            ],
            {
                "0:b": {
                    "class_id": 0,
                    "observations": [
                        parallel_tracking._TrackObservation(
                            t_sec=289.1,
                            frame_idx=2891,
                            bbox_xyxy=[25, 25, 85, 85],
                            confidence=0.95,
                            embedding=None,
                        ),
                        parallel_tracking._TrackObservation(
                            t_sec=289.2,
                            frame_idx=2892,
                            bbox_xyxy=[30, 30, 90, 90],
                            confidence=0.95,
                            embedding=None,
                        ),
                    ],
                }
            },
            {0: "person"},
        )

    monkeypatch.setattr(
        parallel_tracking,
        "_extract_chunk_local_tracks",
        _fake_extract_chunk_local_tracks,
    )

    payload = parallel_tracking.run_parallel_chunked_tracking(
        video_path="/tmp/video.mp4",
        settings=settings,
        scenes=[(0.0, 10.0)],
        output_dir=str(tmp_path),
    )

    assert completion_order == [1, 0]
    assert payload["stats"]["row_count"] == 3
    assert payload["entities"][0]["entity_id"] == "person-1"
    assert payload["entities"][0]["appearance_ranges_ms"] == [
        {"start_ms": 289000, "end_ms": 289200}
    ]


def test_repeated_runs_produce_identical_video_summary_payload(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        enable_parallel_chunked_tracking_pipeline=True,
        parallel_tracking_use_clip_embeddings=False,
        parallel_tracking_output_mode="summary_v2",
        parallel_tracking_chunk_duration_sec=300,
        parallel_tracking_overlap_sec=15,
        parallel_tracking_sample_fps=10,
        parallel_tracking_chunk_max_workers=2,
        parallel_tracking_min_cosine=0.0,
        parallel_tracking_min_iou=0.0,
    )

    class _DummyCap:
        def __init__(self, _video_path: str) -> None:
            pass

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 1000.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 900.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 900.0
            return 0.0

        def release(self) -> None:
            return None

    monkeypatch.setattr(parallel_tracking.cv2, "VideoCapture", _DummyCap)
    monkeypatch.setattr(
        parallel_tracking,
        "_build_embedder",
        lambda _config: (parallel_tracking._FallbackAppearanceEmbedder(), "fallback"),
    )
    monkeypatch.setattr(
        parallel_tracking,
        "_plan_chunks",
        lambda *_args, **_kwargs: [
            parallel_tracking._ChunkWindow(chunk_id=0, start_sec=0.0, end_sec=30.0),
        ],
    )
    monkeypatch.setattr(
        parallel_tracking,
        "_extract_chunk_local_tracks",
        lambda **_kwargs: (
            [
                {
                    "t_sec": 1.0,
                    "frame_idx": 10,
                    "local_id": "0:a",
                    "class_id": 0,
                    "conf": 0.9,
                    "bbox_xyxy": [20, 20, 80, 80],
                },
                {
                    "t_sec": 2.0,
                    "frame_idx": 20,
                    "local_id": "0:a",
                    "class_id": 0,
                    "conf": 0.9,
                    "bbox_xyxy": [200, 200, 260, 260],
                },
            ],
            {
                "0:a": {
                    "class_id": 0,
                    "observations": [
                        parallel_tracking._TrackObservation(
                            t_sec=1.0,
                            frame_idx=10,
                            bbox_xyxy=[20, 20, 80, 80],
                            confidence=0.9,
                            embedding=None,
                        ),
                        parallel_tracking._TrackObservation(
                            t_sec=2.0,
                            frame_idx=20,
                            bbox_xyxy=[200, 200, 260, 260],
                            confidence=0.9,
                            embedding=None,
                        ),
                    ],
                }
            },
            {0: "person"},
        ),
    )

    first = parallel_tracking.run_parallel_chunked_tracking(
        video_path="/tmp/video.mp4",
        settings=settings,
        scenes=[(0.0, 10.0)],
        output_dir=str(tmp_path / "a"),
    )
    second = parallel_tracking.run_parallel_chunked_tracking(
        video_path="/tmp/video.mp4",
        settings=settings,
        scenes=[(0.0, 10.0)],
        output_dir=str(tmp_path / "b"),
    )

    assert first["zone_definition"] == second["zone_definition"]
    assert first["entities"] == second["entities"]


def test_sqlite_ground_truth_persistence_round_trip(tmp_path):
    rows = [
        _row(t_ms=1000, global_id=1, class_id=0, conf=0.7, x1=10, y1=20, x2=30, y2=40),
        _row(t_ms=1100, global_id=1, class_id=0, conf=0.8, x1=11, y1=21, x2=31, y2=41),
    ]
    sqlite_path = tmp_path / "tracks.canonical.sqlite3"
    parallel_tracking._persist_canonical_rows_sqlite(rows=rows, db_path=sqlite_path)
    restored = parallel_tracking._read_canonical_rows_sqlite(db_path=sqlite_path)

    assert restored == rows
