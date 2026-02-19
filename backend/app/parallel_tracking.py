"""Parallel chunked object tracking with cross-chunk ID stitching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_EPS = 1e-6
_INVALID_ASSIGNMENT_COST = 1_000_000.0
_DEFAULT_METHOD = "chunked_botsort_stitch_v1"
_KEYBOX_IOU_THRESHOLD = 0.85
_KEYBOX_CENTER_DELTA_RATIO = 0.02
_KEYBOX_AREA_RATIO_DELTA = 0.15
_KEYBOX_MAX_GAP_SEC = 1.0
_MIN_TRACK_SPAN_DURATION_SEC = 0.7
_MIN_TRACK_KEYBOX_COUNT = 3
_MIN_TRACK_AVG_AREA_RATIO = 0.0005
_NORMALIZATION_SCALE = 10_000
_CANONICAL_TABLE_NAME = "chunked_track_rows"
_SUPPORTED_GROUND_TRUTH_BACKENDS = frozenset({"sqlite", "parquet"})
_SUPPORTED_OUTPUT_MODES = frozenset({"legacy", "summary_v2", "dual"})
_DEFAULT_GROUND_TRUTH_BACKEND = "sqlite"
_DEFAULT_OUTPUT_MODE = "summary_v2"
_DEFAULT_GRID_WIDTH = 12
_DEFAULT_GRID_HEIGHT = 7
_DEFAULT_TOP_ENTITIES = 30
_DEFAULT_MAX_PATH_SEGMENTS = 6
_DEFAULT_EVIDENCE_POINTS = 2
_MOTION_STATIONARY_THRESHOLD = 0.003
_MOTION_MOVING_THRESHOLD = 0.02
_DEFAULT_BACKEND_STRATEGY = "default"
_DEFAULT_STITCH_STRATEGY = "default"
_DEFAULT_ZONE_STRATEGY = "grid3x3"
_ZONE_LABELS_3X3: tuple[str, ...] = (
    "top-left",
    "top-center",
    "top-right",
    "middle-left",
    "center",
    "middle-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)
_PERSON_LABELS = frozenset({"person"})


@dataclass(frozen=True)
class ChunkedTrackingConfig:
    """Runtime knobs for parallel chunked tracking."""

    enabled: bool
    chunk_duration_sec: float
    overlap_sec: float
    sample_fps: int
    max_entities: int
    chunk_max_workers: int
    detector_weights: str
    tracker_config: str
    backend_strategy: str
    stitch_strategy: str
    zone_strategy: str
    confidence_threshold: float
    min_cosine: float
    min_iou: float
    velocity_window: int
    use_clip_embeddings: bool
    clip_model_id: str
    clip_pretrained: str
    ground_truth_backend: str
    output_mode: str
    top_entities_per_scene: int
    path_grid_width: int
    path_grid_height: int
    path_max_segments: int

    @classmethod
    def from_settings(cls, settings: Any) -> "ChunkedTrackingConfig":
        chunk_duration = max(
            30.0, float(getattr(settings, "parallel_tracking_chunk_duration_sec", 300.0))
        )
        overlap = max(
            1.0,
            min(
                chunk_duration - 1.0,
                float(getattr(settings, "parallel_tracking_overlap_sec", 15.0)),
            ),
        )
        sample_fps = max(1, int(getattr(settings, "parallel_tracking_sample_fps", 10)))
        max_entities = max(
            1, int(getattr(settings, "parallel_tracking_max_entities", 20))
        )
        chunk_max_workers = max(
            1, int(getattr(settings, "parallel_tracking_chunk_max_workers", 1))
        )
        confidence = float(
            getattr(settings, "parallel_tracking_confidence_threshold", 0.05)
        )
        min_cosine = float(getattr(settings, "parallel_tracking_min_cosine", 0.30))
        min_iou = float(getattr(settings, "parallel_tracking_min_iou", 0.10))
        velocity_window = max(
            2, int(getattr(settings, "parallel_tracking_velocity_window", 3))
        )
        return cls(
            enabled=bool(
                getattr(settings, "enable_parallel_chunked_tracking_pipeline", False)
            ),
            chunk_duration_sec=chunk_duration,
            overlap_sec=overlap,
            sample_fps=sample_fps,
            max_entities=max_entities,
            chunk_max_workers=chunk_max_workers,
            detector_weights=str(
                getattr(settings, "parallel_tracking_detector_weights", "yolo11n.pt")
            ),
            tracker_config=str(
                getattr(settings, "parallel_tracking_tracker_config", "botsort_reid.yaml")
            ),
            backend_strategy=str(
                getattr(
                    settings,
                    "parallel_tracking_backend_strategy",
                    _DEFAULT_BACKEND_STRATEGY,
                )
            )
            .strip()
            .lower(),
            stitch_strategy=str(
                getattr(
                    settings,
                    "parallel_tracking_stitch_strategy",
                    _DEFAULT_STITCH_STRATEGY,
                )
            )
            .strip()
            .lower(),
            zone_strategy=str(
                getattr(
                    settings,
                    "parallel_tracking_zone_strategy",
                    _DEFAULT_ZONE_STRATEGY,
                )
            )
            .strip()
            .lower(),
            confidence_threshold=max(0.0, min(confidence, 1.0)),
            min_cosine=max(-1.0, min(min_cosine, 1.0)),
            min_iou=max(0.0, min(min_iou, 1.0)),
            velocity_window=velocity_window,
            use_clip_embeddings=bool(
                getattr(settings, "parallel_tracking_use_clip_embeddings", True)
            ),
            clip_model_id=str(
                getattr(settings, "parallel_tracking_clip_model_id", "ViT-B-32")
            ),
            clip_pretrained=str(
                getattr(settings, "parallel_tracking_clip_pretrained", "openai")
            ),
            ground_truth_backend=str(
                getattr(
                    settings,
                    "parallel_tracking_ground_truth_backend",
                    _DEFAULT_GROUND_TRUTH_BACKEND,
                )
            )
            .strip()
            .lower(),
            output_mode=str(
                getattr(
                    settings,
                    "parallel_tracking_output_mode",
                    _DEFAULT_OUTPUT_MODE,
                )
            )
            .strip()
            .lower(),
            top_entities_per_scene=max(
                1,
                int(
                    getattr(
                        settings,
                        "parallel_tracking_scene_top_entities",
                        _DEFAULT_TOP_ENTITIES,
                    )
                ),
            ),
            path_grid_width=max(
                1,
                int(
                    getattr(
                        settings,
                        "parallel_tracking_scene_grid_width",
                        _DEFAULT_GRID_WIDTH,
                    )
                ),
            ),
            path_grid_height=max(
                1,
                int(
                    getattr(
                        settings,
                        "parallel_tracking_scene_grid_height",
                        _DEFAULT_GRID_HEIGHT,
                    )
                ),
            ),
            path_max_segments=max(
                1,
                int(
                    getattr(
                        settings,
                        "parallel_tracking_scene_path_max_segments",
                        _DEFAULT_MAX_PATH_SEGMENTS,
                    )
                ),
            ),
        )

    def normalized_ground_truth_backend(self) -> str:
        backend = self.ground_truth_backend
        if backend in _SUPPORTED_GROUND_TRUTH_BACKENDS:
            return backend
        return _DEFAULT_GROUND_TRUTH_BACKEND

    def normalized_output_mode(self) -> str:
        mode = self.output_mode
        if mode in _SUPPORTED_OUTPUT_MODES:
            return mode
        return _DEFAULT_OUTPUT_MODE

    def normalized_backend_strategy(self) -> str:
        return self.backend_strategy or _DEFAULT_BACKEND_STRATEGY

    def normalized_stitch_strategy(self) -> str:
        return self.stitch_strategy or _DEFAULT_STITCH_STRATEGY

    def normalized_zone_strategy(self) -> str:
        return self.zone_strategy or _DEFAULT_ZONE_STRATEGY


@dataclass(frozen=True)
class CanonicalTrackRow:
    """Canonical stitched per-frame object track row."""

    t_ms: int
    global_id: int
    class_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class _ChunkWindow:
    chunk_id: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class _TrackObservation:
    t_sec: float
    frame_idx: int
    bbox_xyxy: list[int]
    confidence: float
    embedding: np.ndarray | None


@dataclass(frozen=True)
class _TrackSignature:
    local_id: str
    class_id: int
    start_t: float
    end_t: float
    bbox_start: list[int]
    bbox_end: list[int]
    velocity_xy: tuple[float, float]
    embedding: np.ndarray | None


@dataclass(frozen=True)
class _GlobalTrackState:
    global_id: str
    class_id: int
    embedding: np.ndarray | None
    bbox_end: list[int]
    velocity_xy: tuple[float, float]
    last_t: float


class _FallbackAppearanceEmbedder:
    """Geometry/color fallback used when CLIP embeddings are unavailable."""

    def embed(self, image: np.ndarray, bbox_xyxy: list[int]) -> np.ndarray | None:
        crop = _extract_crop(image, bbox_xyxy)
        if crop is None:
            return None
        crop_f = crop.astype(np.float32)
        mean_rgb = crop_f.mean(axis=(0, 1))
        std_rgb = crop_f.std(axis=(0, 1))
        height, width = crop_f.shape[:2]
        image_h, image_w = image.shape[:2]
        area_ratio = (width * height) / max(1.0, float(image_h * image_w))
        features = np.array(
            [
                float(mean_rgb[2]),
                float(mean_rgb[1]),
                float(mean_rgb[0]),
                float(std_rgb[2]),
                float(std_rgb[1]),
                float(std_rgb[0]),
                float(width) / max(1.0, float(height)),
                area_ratio,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(features))
        if norm <= _EPS:
            return None
        return features / norm


class _ClipAppearanceEmbedder:
    """OpenCLIP embedder for generic object crops in overlap windows."""

    def __init__(
        self,
        *,
        model_id: str,
        pretrained: str,
    ) -> None:
        import open_clip  # type: ignore[import-not-found]
        import torch

        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_id,
            pretrained=pretrained,
            device=device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        logger.info(
            "parallel_tracking.clip_embedder enabled model=%s pretrained=%s device=%s",
            model_id,
            pretrained,
            device,
        )

    def embed(self, image: np.ndarray, bbox_xyxy: list[int]) -> np.ndarray | None:
        crop = _extract_crop(image, bbox_xyxy)
        if crop is None:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._preprocess(pil).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            encoded = self._model.encode_image(tensor)
        vector = encoded[0].detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= _EPS:
            return None
        return vector / norm


def _build_embedder(config: ChunkedTrackingConfig) -> tuple[Any, str]:
    if not config.use_clip_embeddings:
        return _FallbackAppearanceEmbedder(), "fallback"
    try:
        embedder = _ClipAppearanceEmbedder(
            model_id=config.clip_model_id,
            pretrained=config.clip_pretrained,
        )
        return embedder, "clip"
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning(
            "parallel_tracking.clip_embedder unavailable, using fallback: %s",
            exc,
        )
        return _FallbackAppearanceEmbedder(), "fallback"


def _extract_crop(image: np.ndarray, bbox_xyxy: list[int]) -> np.ndarray | None:
    if image.size == 0:
        return None
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(bbox_xyxy[0])))
    y1 = max(0, min(height - 1, int(bbox_xyxy[1])))
    x2 = max(0, min(width, int(bbox_xyxy[2])))
    y2 = max(0, min(height, int(bbox_xyxy[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _center_xy(bbox_xyxy: list[int]) -> tuple[float, float]:
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area <= 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return float(inter_area) / float(denom)


def _predict_bbox(
    bbox_xyxy: list[int],
    velocity_xy: tuple[float, float],
    dt_sec: float,
) -> list[int]:
    dx = velocity_xy[0] * dt_sec
    dy = velocity_xy[1] * dt_sec
    return [
        int(round(float(bbox_xyxy[0]) + dx)),
        int(round(float(bbox_xyxy[1]) + dy)),
        int(round(float(bbox_xyxy[2]) + dx)),
        int(round(float(bbox_xyxy[3]) + dy)),
    ]


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= _EPS or b_norm <= _EPS:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _sanitize_bbox(raw_bbox: Any) -> list[int] | None:
    if raw_bbox is None:
        return None
    try:
        values = [int(round(float(v))) for v in raw_bbox]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return values


def _plan_chunks(
    duration_sec: float,
    chunk_duration_sec: float,
    overlap_sec: float,
) -> list[_ChunkWindow]:
    if duration_sec <= 0.0:
        return []
    step = max(1.0, chunk_duration_sec - overlap_sec)
    chunks: list[_ChunkWindow] = []
    cursor = 0.0
    chunk_id = 0
    while cursor < duration_sec:
        end_sec = min(duration_sec, cursor + chunk_duration_sec)
        chunks.append(_ChunkWindow(chunk_id=chunk_id, start_sec=cursor, end_sec=end_sec))
        if end_sec >= duration_sec - _EPS:
            break
        cursor += step
        chunk_id += 1
    return chunks


def _sample_timestamps(start_sec: float, end_sec: float, sample_fps: int) -> list[float]:
    if end_sec < start_sec:
        return []
    step = 1.0 / float(max(1, sample_fps))
    samples: list[float] = []
    cursor = float(start_sec)
    while cursor <= end_sec + _EPS:
        samples.append(cursor)
        cursor += step
    return samples


def _resolve_tracker_config(tracker_config: str) -> str:
    raw = str(tracker_config).strip()
    if not raw:
        raw = "botsort_reid.yaml"
    explicit = Path(raw)
    if explicit.is_file():
        return str(explicit)
    local = Path(__file__).resolve().parent / "tracking" / raw
    if local.is_file():
        return str(local)
    return raw


def _estimate_velocity(
    observations: list[_TrackObservation],
    velocity_window: int,
) -> tuple[float, float]:
    if len(observations) < 2:
        return (0.0, 0.0)
    window = observations[-velocity_window:]
    first = window[0]
    last = window[-1]
    dt = max(_EPS, last.t_sec - first.t_sec)
    first_cx, first_cy = _center_xy(first.bbox_xyxy)
    last_cx, last_cy = _center_xy(last.bbox_xyxy)
    return ((last_cx - first_cx) / dt, (last_cy - first_cy) / dt)


def _mean_embedding(vectors: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [vector for vector in vectors if vector is not None]
    if not valid:
        return None
    stacked = np.stack(valid, axis=0)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= _EPS:
        return None
    return (mean / norm).astype(np.float32)


def _build_track_signatures(
    local_tracks: dict[str, dict[str, Any]],
    *,
    window_start_sec: float,
    window_end_sec: float,
    velocity_window: int,
) -> dict[str, _TrackSignature]:
    signatures: dict[str, _TrackSignature] = {}
    for local_id, slot in local_tracks.items():
        class_id = int(slot["class_id"])
        observations = [
            obs
            for obs in slot["observations"]
            if window_start_sec - _EPS <= obs.t_sec <= window_end_sec + _EPS
        ]
        if not observations:
            continue
        observations.sort(key=lambda item: (item.t_sec, item.frame_idx))
        first = observations[0]
        last = observations[-1]
        velocity_xy = _estimate_velocity(observations, velocity_window)
        sample_indices = sorted({0, len(observations) // 2, len(observations) - 1})
        sample_embeddings = [observations[index].embedding for index in sample_indices]
        embedding = _mean_embedding(sample_embeddings)
        signatures[local_id] = _TrackSignature(
            local_id=local_id,
            class_id=class_id,
            start_t=first.t_sec,
            end_t=last.t_sec,
            bbox_start=list(first.bbox_xyxy),
            bbox_end=list(last.bbox_xyxy),
            velocity_xy=velocity_xy,
            embedding=embedding,
        )
    return signatures


def _run_linear_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-not-found]
    except Exception:
        row_count, col_count = cost_matrix.shape
        assignments: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for cost, row_idx, col_idx in sorted(
            (
                (float(cost_matrix[row_idx, col_idx]), row_idx, col_idx)
                for row_idx in range(row_count)
                for col_idx in range(col_count)
            ),
            key=lambda item: item[0],
        ):
            if row_idx in used_rows or col_idx in used_cols:
                continue
            assignments.append((row_idx, col_idx))
            used_rows.add(row_idx)
            used_cols.add(col_idx)
        return assignments

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return list(zip(row_indices.tolist(), col_indices.tolist(), strict=False))


def _stitch_boundary(
    *,
    active_global_tracks: dict[str, _GlobalTrackState],
    local_head_signatures: dict[str, _TrackSignature],
    min_cosine: float,
    min_iou: float,
    overlap_sec: float,
    next_global_index: int,
) -> tuple[dict[str, str], int]:
    local_ids = sorted(local_head_signatures)
    global_ids = sorted(active_global_tracks)
    mapping: dict[str, str] = {}

    if local_ids and global_ids:
        cost_matrix = np.full(
            (len(global_ids), len(local_ids)),
            _INVALID_ASSIGNMENT_COST,
            dtype=np.float32,
        )
        metrics: dict[tuple[int, int], tuple[float, float]] = {}
        for row_idx, global_id in enumerate(global_ids):
            global_state = active_global_tracks[global_id]
            for col_idx, local_id in enumerate(local_ids):
                local_sig = local_head_signatures[local_id]
                if global_state.class_id != local_sig.class_id:
                    continue
                dt_sec = max(0.0, local_sig.start_t - global_state.last_t)
                if dt_sec > overlap_sec + _EPS:
                    continue
                predicted_bbox = _predict_bbox(
                    global_state.bbox_end,
                    global_state.velocity_xy,
                    dt_sec,
                )
                iou = _bbox_iou(predicted_bbox, local_sig.bbox_start)
                cosine = _cosine_similarity(global_state.embedding, local_sig.embedding)
                score = (0.6 * cosine) + (0.4 * iou)
                cost_matrix[row_idx, col_idx] = -score
                metrics[(row_idx, col_idx)] = (cosine, iou)

        for row_idx, col_idx in _run_linear_assignment(cost_matrix):
            if float(cost_matrix[row_idx, col_idx]) >= _INVALID_ASSIGNMENT_COST:
                continue
            cosine, iou = metrics.get((row_idx, col_idx), (0.0, 0.0))
            if cosine < min_cosine or iou < min_iou:
                continue
            mapping[local_ids[col_idx]] = global_ids[row_idx]

    for local_id in local_ids:
        if local_id in mapping:
            continue
        mapping[local_id] = f"global_{next_global_index}"
        next_global_index += 1
    return mapping, next_global_index


def _extract_chunk_local_tracks(
    *,
    video_path: str,
    chunk: _ChunkWindow,
    config: ChunkedTrackingConfig,
    native_fps: float,
    embedder: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[int, str]]:
    from ultralytics import YOLO

    model = YOLO(config.detector_weights)
    cap = cv2.VideoCapture(video_path)
    rows: list[dict[str, Any]] = []
    local_tracks: dict[str, dict[str, Any]] = {}
    class_name_map: dict[int, str] = {}
    tracker_config = _resolve_tracker_config(config.tracker_config)

    overlap_head_end = min(chunk.end_sec, chunk.start_sec + config.overlap_sec)
    overlap_tail_start = max(chunk.start_sec, chunk.end_sec - config.overlap_sec)

    try:
        for t_sec in _sample_timestamps(chunk.start_sec, chunk.end_sec, config.sample_fps):
            frame_idx = max(0, int(round(t_sec * native_fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                tracker=tracker_config,
                conf=config.confidence_threshold,
            )
            if not results:
                continue
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                continue
            names = getattr(result, "names", {}) or {}
            if isinstance(names, dict):
                for key, value in names.items():
                    try:
                        class_key = int(key)
                    except (TypeError, ValueError):
                        continue
                    class_name_map[class_key] = str(value)

            xyxy = boxes.xyxy.cpu().numpy()
            conf = (
                boxes.conf.cpu().numpy()
                if getattr(boxes, "conf", None) is not None
                else np.zeros(len(xyxy), dtype=np.float32)
            )
            cls = (
                boxes.cls.cpu().numpy()
                if getattr(boxes, "cls", None) is not None
                else np.zeros(len(xyxy), dtype=np.float32)
            )
            box_ids = None
            if hasattr(boxes, "id") and boxes.id is not None:
                try:
                    box_ids = boxes.id.cpu().numpy()
                except Exception:
                    box_ids = None

            for det_index, raw_bbox in enumerate(xyxy):
                bbox_xyxy = _sanitize_bbox(raw_bbox.tolist())
                if bbox_xyxy is None:
                    continue
                class_id = int(cls[det_index]) if det_index < len(cls) else 0
                confidence = float(conf[det_index]) if det_index < len(conf) else 0.0

                raw_id: str
                if box_ids is not None and det_index < len(box_ids):
                    candidate = box_ids[det_index]
                    if candidate is None or (
                        isinstance(candidate, float) and np.isnan(candidate)
                    ):
                        raw_id = f"f{frame_idx}_{det_index + 1}"
                    else:
                        raw_id = str(int(candidate))
                else:
                    raw_id = f"f{frame_idx}_{det_index + 1}"
                local_id = f"{class_id}:{raw_id}"

                is_overlap_sample = (
                    (t_sec <= overlap_head_end + _EPS)
                    or (t_sec >= overlap_tail_start - _EPS)
                )
                embedding = (
                    embedder.embed(frame, bbox_xyxy) if is_overlap_sample else None
                )

                rows.append(
                    {
                        "t_sec": float(t_sec),
                        "frame_idx": frame_idx,
                        "local_id": local_id,
                        "class_id": class_id,
                        "conf": confidence,
                        "bbox_xyxy": bbox_xyxy,
                    }
                )
                slot = local_tracks.setdefault(
                    local_id,
                    {
                        "class_id": class_id,
                        "observations": [],
                    },
                )
                slot["observations"].append(
                    _TrackObservation(
                        t_sec=float(t_sec),
                        frame_idx=frame_idx,
                        bbox_xyxy=list(bbox_xyxy),
                        confidence=confidence,
                        embedding=embedding,
                    )
                )
    finally:
        cap.release()
    return rows, local_tracks, class_name_map


ChunkExtractionResult = (
    tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[int, str]]
    | tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]
)
ChunkExtractionStrategy = Callable[..., ChunkExtractionResult]
StitchStrategy = Callable[
    ...,
    tuple[dict[str, str], int],
]
ZoneDefinitionStrategy = Callable[[int, int], dict[str, Any]]


def _resolve_strategy(
    *,
    strategy_kind: str,
    strategy_id: str,
    registry: dict[str, Callable[..., Any]],
) -> Callable[..., Any]:
    normalized = strategy_id.strip().lower()
    if normalized in registry:
        return registry[normalized]
    available = ", ".join(sorted(registry))
    raise ValueError(
        f"Invalid {strategy_kind} strategy '{strategy_id}'. Available: {available}"
    )


def _global_id_as_int(global_id: str) -> int:
    value = str(global_id)
    if value.startswith("global_"):
        suffix = value.removeprefix("global_")
        if suffix.isdigit():
            return int(suffix)
    if value.isdigit():
        return int(value)
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits:
        return int(digits)
    return 0


def _resolve_output_dir(*, video_path: str, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    source_path = Path(video_path).resolve()
    if source_path.parent.name == "input":
        return source_path.parent.parent / "tracking"
    return source_path.parent / "tracking"


def _normalize_canonical_row(
    *,
    t_ms: Any,
    global_id: Any,
    class_id: Any,
    conf: Any,
    x1: Any,
    y1: Any,
    x2: Any,
    y2: Any,
) -> CanonicalTrackRow:
    return CanonicalTrackRow(
        t_ms=max(0, int(round(float(t_ms)))),
        global_id=max(0, int(global_id)),
        class_id=int(class_id),
        conf=float(conf),
        x1=int(round(float(x1))),
        y1=int(round(float(y1))),
        x2=int(round(float(x2))),
        y2=int(round(float(y2))),
    )


def _persist_canonical_rows_sqlite(*, rows: list[CanonicalTrackRow], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_CANONICAL_TABLE_NAME} (
                t_ms INTEGER NOT NULL,
                global_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                conf REAL NOT NULL,
                x1 INTEGER NOT NULL,
                y1 INTEGER NOT NULL,
                x2 INTEGER NOT NULL,
                y2 INTEGER NOT NULL
            )
            """
        )
        conn.execute(f"DELETE FROM {_CANONICAL_TABLE_NAME}")
        conn.executemany(
            f"""
            INSERT INTO {_CANONICAL_TABLE_NAME}
            (t_ms, global_id, class_id, conf, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(row.t_ms),
                    int(row.global_id),
                    int(row.class_id),
                    float(row.conf),
                    int(row.x1),
                    int(row.y1),
                    int(row.x2),
                    int(row.y2),
                )
                for row in rows
            ],
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{_CANONICAL_TABLE_NAME}_gid_t
            ON {_CANONICAL_TABLE_NAME} (global_id, t_ms)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{_CANONICAL_TABLE_NAME}_class_t
            ON {_CANONICAL_TABLE_NAME} (class_id, t_ms)
            """
        )
        conn.commit()


def _read_canonical_rows_sqlite(*, db_path: Path) -> list[CanonicalTrackRow]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            f"""
            SELECT t_ms, global_id, class_id, conf, x1, y1, x2, y2
            FROM {_CANONICAL_TABLE_NAME}
            ORDER BY global_id ASC, t_ms ASC
            """
        )
        rows = cursor.fetchall()
    return [
        _normalize_canonical_row(
            t_ms=t_ms,
            global_id=global_id,
            class_id=class_id,
            conf=conf,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        for t_ms, global_id, class_id, conf, x1, y1, x2, y2 in rows
    ]


def _persist_canonical_rows_parquet(*, rows: list[CanonicalTrackRow], parquet_path: Path) -> None:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Parquet backend requires 'pyarrow'. Set PARALLEL_TRACKING_GROUND_TRUTH_BACKEND=sqlite or install pyarrow."
        ) from exc

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("t_ms", pa.int64()),
            ("global_id", pa.int64()),
            ("class_id", pa.int64()),
            ("conf", pa.float32()),
            ("x1", pa.int32()),
            ("y1", pa.int32()),
            ("x2", pa.int32()),
            ("y2", pa.int32()),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {
                "t_ms": int(row.t_ms),
                "global_id": int(row.global_id),
                "class_id": int(row.class_id),
                "conf": float(row.conf),
                "x1": int(row.x1),
                "y1": int(row.y1),
                "x2": int(row.x2),
                "y2": int(row.y2),
            }
            for row in rows
        ],
        schema=schema,
    )
    pq.write_table(table, parquet_path)


def _read_canonical_rows_parquet(*, parquet_path: Path) -> list[CanonicalTrackRow]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Parquet backend requires 'pyarrow'. Set PARALLEL_TRACKING_GROUND_TRUTH_BACKEND=sqlite or install pyarrow."
        ) from exc

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    canonical = [
        _normalize_canonical_row(
            t_ms=row.get("t_ms", 0),
            global_id=row.get("global_id", 0),
            class_id=row.get("class_id", 0),
            conf=row.get("conf", 0.0),
            x1=row.get("x1", 0),
            y1=row.get("y1", 0),
            x2=row.get("x2", 0),
            y2=row.get("y2", 0),
        )
        for row in rows
    ]
    return sorted(canonical, key=lambda item: (item.global_id, item.t_ms, item.class_id))


def _persist_and_reload_canonical_rows(
    *,
    rows: list[CanonicalTrackRow],
    backend: str,
    output_root: Path,
) -> tuple[list[CanonicalTrackRow], dict[str, str]]:
    normalized_backend = (
        backend if backend in _SUPPORTED_GROUND_TRUTH_BACKENDS else _DEFAULT_GROUND_TRUTH_BACKEND
    )
    if normalized_backend == "parquet":
        parquet_path = output_root / "tracks.canonical.parquet"
        _persist_canonical_rows_parquet(rows=rows, parquet_path=parquet_path)
        return _read_canonical_rows_parquet(parquet_path=parquet_path), {
            "canonical_parquet": str(parquet_path)
        }

    sqlite_path = output_root / "tracks.canonical.sqlite3"
    _persist_canonical_rows_sqlite(rows=rows, db_path=sqlite_path)
    return _read_canonical_rows_sqlite(db_path=sqlite_path), {
        "canonical_sqlite": str(sqlite_path)
    }


def _split_track_spans(
    *,
    rows: list[CanonicalTrackRow],
    sample_fps: int,
) -> list[list[CanonicalTrackRow]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: int(item.t_ms))
    gap_threshold = 1.5 / max(1, sample_fps)
    spans: list[list[CanonicalTrackRow]] = []
    current: list[CanonicalTrackRow] = [ordered[0]]
    for row in ordered[1:]:
        dt = (float(row.t_ms) - float(current[-1].t_ms)) / 1000.0
        if dt > gap_threshold + _EPS:
            spans.append(current)
            current = [row]
            continue
        current.append(row)
    spans.append(current)
    return spans


def _bbox_area(box: list[int]) -> float:
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    return width * height


def _normalize_coord(value: float, denom: float) -> int:
    denominator = max(1.0, denom)
    scaled = int(round((float(value) / denominator) * _NORMALIZATION_SCALE))
    return max(0, min(_NORMALIZATION_SCALE, scaled))


def _encode_key_box(
    *,
    row: CanonicalTrackRow,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    x1 = int(row.x1)
    y1 = int(row.y1)
    x2 = int(row.x2)
    y2 = int(row.y2)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    return [
        int(row.t_ms),
        _normalize_coord(center_x, float(frame_width)),
        _normalize_coord(center_y, float(frame_height)),
        _normalize_coord(float(box_w), float(frame_width)),
        _normalize_coord(float(box_h), float(frame_height)),
    ]


def _select_key_boxes(
    *,
    span_rows: list[CanonicalTrackRow],
    frame_width: int,
    frame_height: int,
) -> list[list[int]]:
    if not span_rows:
        return []
    ordered = sorted(span_rows, key=lambda item: int(item.t_ms))
    if len(ordered) == 1:
        return [
            _encode_key_box(
                row=ordered[0], frame_width=frame_width, frame_height=frame_height
            )
        ]

    kept_rows: list[CanonicalTrackRow] = [ordered[0]]
    last_kept = ordered[0]
    center_threshold = _KEYBOX_CENTER_DELTA_RATIO * float(
        min(frame_width, frame_height)
    )

    for row in ordered[1:-1]:
        current_box = [int(row.x1), int(row.y1), int(row.x2), int(row.y2)]
        last_box = [
            int(last_kept.x1),
            int(last_kept.y1),
            int(last_kept.x2),
            int(last_kept.y2),
        ]
        iou = _bbox_iou(last_box, current_box)
        current_center = _center_xy(current_box)
        last_center = _center_xy(last_box)
        center_delta = float(
            np.hypot(
                current_center[0] - last_center[0],
                current_center[1] - last_center[1],
            )
        )
        area_ratio = _bbox_area(current_box) / max(_EPS, _bbox_area(last_box))
        time_delta = (float(row.t_ms) - float(last_kept.t_ms)) / 1000.0
        should_keep = (
            iou < _KEYBOX_IOU_THRESHOLD
            or center_delta > center_threshold
            or abs(area_ratio - 1.0) > _KEYBOX_AREA_RATIO_DELTA
            or time_delta >= _KEYBOX_MAX_GAP_SEC
        )
        if should_keep:
            kept_rows.append(row)
            last_kept = row

    if kept_rows[-1] is not ordered[-1]:
        kept_rows.append(ordered[-1])

    return [
        _encode_key_box(row=row, frame_width=frame_width, frame_height=frame_height)
        for row in kept_rows
    ]


def _build_compact_tracks(
    *,
    canonical_rows: list[CanonicalTrackRow],
    sample_fps: int,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[CanonicalTrackRow]] = {}
    for row in canonical_rows:
        key = (int(row.global_id), int(row.class_id))
        grouped.setdefault(key, []).append(row)

    frame_area = max(1.0, float(frame_width * frame_height))
    tracks: list[dict[str, Any]] = []

    for (global_id, class_id), rows in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(rows, key=lambda item: int(item.t_ms))
        area_ratios = []
        for row in ordered:
            box = [int(row.x1), int(row.y1), int(row.x2), int(row.y2)]
            area_ratios.append(_bbox_area(box) / frame_area)
        average_area_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0
        if average_area_ratio < _MIN_TRACK_AVG_AREA_RATIO:
            continue

        spans_rows = _split_track_spans(rows=ordered, sample_fps=sample_fps)
        compact_spans: list[dict[str, Any]] = []
        max_span_duration_sec = 0.0
        total_key_boxes = 0
        for span_rows in spans_rows:
            key_boxes = _select_key_boxes(
                span_rows=span_rows,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if not key_boxes:
                continue
            start_ms = int(span_rows[0].t_ms)
            end_ms = int(span_rows[-1].t_ms)
            compact_spans.append(
                {
                    "s_ms": start_ms,
                    "e_ms": end_ms,
                    "k": key_boxes,
                }
            )
            max_span_duration_sec = max(
                max_span_duration_sec, max(0.0, float(end_ms - start_ms) / 1000.0)
            )
            total_key_boxes += len(key_boxes)

        keep_track = (
            max_span_duration_sec >= _MIN_TRACK_SPAN_DURATION_SEC
            or total_key_boxes >= _MIN_TRACK_KEYBOX_COUNT
        )
        if not keep_track or not compact_spans:
            continue

        confidence = (
            float(np.mean([float(row.conf) for row in ordered])) if ordered else 0.0
        )
        tracks.append(
            {
                "id": global_id,
                "cls": class_id,
                "conf": round(confidence, 6),
                "spans": compact_spans,
            }
        )

    return sorted(tracks, key=lambda item: int(item["id"]))


def _build_scene_packets(
    *,
    tracks: list[dict[str, Any]],
    scenes: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    if not scenes:
        return []
    packets: list[dict[str, Any]] = []
    for scene_start_sec, scene_end_sec in scenes:
        scene_ts_ms = max(0, int(round(float(scene_start_sec) * 1000.0)))
        scene_te_ms = max(scene_ts_ms, int(round(float(scene_end_sec) * 1000.0)))

        track_ids: list[int] = []
        track_slices: list[dict[str, Any]] = []
        for track in tracks:
            clipped: list[list[int]] = []
            for span in track.get("spans", []):
                key_boxes = span.get("k", [])
                if not isinstance(key_boxes, list):
                    continue
                for key_box in key_boxes:
                    if not isinstance(key_box, list) or len(key_box) != 5:
                        continue
                    t_ms = int(key_box[0])
                    if scene_ts_ms <= t_ms <= scene_te_ms:
                        clipped.append([int(value) for value in key_box])
            if not clipped:
                continue
            track_id = int(track["id"])
            track_ids.append(track_id)
            track_slices.append(
                {
                    "id": track_id,
                    "cls": int(track["cls"]),
                    "k": clipped,
                }
            )

        track_ids.sort()
        track_slices.sort(key=lambda item: int(item["id"]))
        packets.append(
            {
                "scene_ts_ms": scene_ts_ms,
                "scene_te_ms": scene_te_ms,
                "track_ids": track_ids,
                "track_slices": track_slices,
            }
        )
    return packets


def _row_bbox(row: CanonicalTrackRow) -> list[int]:
    return [int(row.x1), int(row.y1), int(row.x2), int(row.y2)]


def _row_center(row: CanonicalTrackRow) -> tuple[float, float]:
    return _center_xy(_row_bbox(row))


def _row_area(row: CanonicalTrackRow) -> float:
    return _bbox_area(_row_bbox(row))


def _classify_motion(
    *,
    rows: list[CanonicalTrackRow],
    frame_diagonal: float,
) -> str:
    if len(rows) < 2:
        return "S"
    speeds: list[float] = []
    ordered = sorted(rows, key=lambda item: int(item.t_ms))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        dt_sec = max(_EPS, (float(current.t_ms) - float(previous.t_ms)) / 1000.0)
        prev_x, prev_y = _row_center(previous)
        cur_x, cur_y = _row_center(current)
        pixels_per_second = float(np.hypot(cur_x - prev_x, cur_y - prev_y)) / dt_sec
        speeds.append(pixels_per_second / max(1.0, frame_diagonal))
    if not speeds:
        return "S"
    median_speed = float(np.median(speeds))
    if median_speed < _MOTION_STATIONARY_THRESHOLD:
        return "S"
    if median_speed < _MOTION_MOVING_THRESHOLD:
        return "M"
    return "F"


def _grid_cell_id(
    *,
    row: CanonicalTrackRow,
    frame_width: int,
    frame_height: int,
    grid_width: int,
    grid_height: int,
) -> int:
    cx, cy = _row_center(row)
    width = max(1.0, float(frame_width))
    height = max(1.0, float(frame_height))
    gx = int((cx / width) * grid_width)
    gy = int((cy / height) * grid_height)
    gx = max(0, min(grid_width - 1, gx))
    gy = max(0, min(grid_height - 1, gy))
    return (gy * grid_width) + gx


def _build_rle_path_segments(
    *,
    rows: list[CanonicalTrackRow],
    frame_width: int,
    frame_height: int,
    grid_width: int,
    grid_height: int,
) -> list[dict[str, int]]:
    ordered = sorted(rows, key=lambda item: int(item.t_ms))
    segments: list[dict[str, int]] = []
    for row in ordered:
        cell_id = _grid_cell_id(
            row=row,
            frame_width=frame_width,
            frame_height=frame_height,
            grid_width=grid_width,
            grid_height=grid_height,
        )
        t_ms = int(row.t_ms)
        if not segments or segments[-1]["cell_id"] != cell_id:
            segments.append({"cell_id": cell_id, "start_ms": t_ms, "end_ms": t_ms})
            continue
        segments[-1]["end_ms"] = t_ms
    return segments


def _merge_path_segments_to_cap(
    *,
    segments: list[dict[str, int]],
    max_segments: int,
) -> list[dict[str, int]]:
    if len(segments) <= max_segments:
        return segments

    merged = [dict(segment) for segment in segments]

    def _segment_duration(segment: dict[str, int]) -> int:
        return max(1, int(segment["end_ms"]) - int(segment["start_ms"]))

    while len(merged) > max_segments:
        shortest_index = min(
            range(len(merged)),
            key=lambda idx: (_segment_duration(merged[idx]), idx),
        )
        if shortest_index == 0:
            neighbor_index = 1
        elif shortest_index == len(merged) - 1:
            neighbor_index = shortest_index - 1
        else:
            left_index = shortest_index - 1
            right_index = shortest_index + 1
            left_duration = _segment_duration(merged[left_index])
            right_duration = _segment_duration(merged[right_index])
            neighbor_index = left_index if left_duration <= right_duration else right_index

        if neighbor_index < shortest_index:
            merged[neighbor_index]["end_ms"] = max(
                int(merged[neighbor_index]["end_ms"]),
                int(merged[shortest_index]["end_ms"]),
            )
            del merged[shortest_index]
            continue

        merged[neighbor_index]["start_ms"] = min(
            int(merged[neighbor_index]["start_ms"]),
            int(merged[shortest_index]["start_ms"]),
        )
        del merged[shortest_index]

    return merged


def _encode_entity_path(
    *,
    rows: list[CanonicalTrackRow],
    frame_width: int,
    frame_height: int,
    grid_width: int,
    grid_height: int,
    max_segments: int,
) -> str | None:
    segments = _build_rle_path_segments(
        rows=rows,
        frame_width=frame_width,
        frame_height=frame_height,
        grid_width=grid_width,
        grid_height=grid_height,
    )
    unique_cell_count = len({int(segment["cell_id"]) for segment in segments})
    if unique_cell_count <= 1:
        return None
    merged = _merge_path_segments_to_cap(segments=segments, max_segments=max_segments)
    return ",".join(
        f"c{int(segment['cell_id'])}@{int(segment['start_ms'])}-{int(segment['end_ms'])}"
        for segment in merged
    )


def _scene_summary_from_rows(
    *,
    canonical_rows: list[CanonicalTrackRow],
    scenes: list[tuple[float, float]],
    class_name_map: dict[int, str],
    frame_width: int,
    frame_height: int,
    config: ChunkedTrackingConfig,
) -> list[dict[str, Any]]:
    if not scenes:
        return []

    frame_diagonal = max(1.0, float(np.hypot(frame_width, frame_height)))
    summaries: list[dict[str, Any]] = []
    top_n = max(1, config.top_entities_per_scene)
    grid_label = f"{config.path_grid_width}x{config.path_grid_height}"

    for scene_id, (scene_start_sec, scene_end_sec) in enumerate(scenes):
        ts_ms = max(0, int(round(float(scene_start_sec) * 1000.0)))
        te_ms = max(ts_ms, int(round(float(scene_end_sec) * 1000.0)))
        scene_rows = [
            row for row in canonical_rows if ts_ms <= int(row.t_ms) <= te_ms
        ]

        by_global_id: dict[int, list[CanonicalTrackRow]] = {}
        for row in scene_rows:
            by_global_id.setdefault(int(row.global_id), []).append(row)

        scored_entities: list[tuple[float, int, str, dict[str, Any]]] = []
        for global_id, rows in sorted(by_global_id.items(), key=lambda item: item[0]):
            ordered = sorted(rows, key=lambda item: int(item.t_ms))
            first_seen = int(ordered[0].t_ms)
            last_seen = int(ordered[-1].t_ms)
            screen_time_ms = max(1, last_seen - first_seen)
            areas = [_row_area(row) for row in ordered]
            median_area = float(np.median(areas)) if areas else 0.0
            score = float(screen_time_ms) * median_area

            class_id = int(ordered[0].class_id)
            label = class_name_map.get(class_id, f"class_{class_id}")
            motion = _classify_motion(rows=ordered, frame_diagonal=frame_diagonal)

            best = max(ordered, key=lambda row: (_row_area(row), -int(row.t_ms)))
            path = _encode_entity_path(
                rows=ordered,
                frame_width=frame_width,
                frame_height=frame_height,
                grid_width=config.path_grid_width,
                grid_height=config.path_grid_height,
                max_segments=config.path_max_segments,
            )

            entity: dict[str, Any] = {
                "id": int(global_id),
                "label": label,
                "p": [first_seen, last_seen],
                "m": motion,
                "ev": [int(best.t_ms), last_seen][: _DEFAULT_EVIDENCE_POINTS],
            }
            if path:
                entity["path"] = path
            scored_entities.append((score, int(global_id), label, entity))

        scored_entities.sort(key=lambda item: (-item[0], item[1]))
        top_entities = [item[3] for item in scored_entities[:top_n]]
        tail_counts: dict[str, int] = {}
        for _, _, label, _ in scored_entities[top_n:]:
            tail_counts[label] = tail_counts.get(label, 0) + 1

        summaries.append(
            {
                "scene_id": scene_id,
                "ts_ms": ts_ms,
                "te_ms": te_ms,
                "grid": grid_label,
                "entities_top": top_entities,
                "counts_by_label_tail": {
                    key: tail_counts[key] for key in sorted(tail_counts)
                },
            }
        )
    return summaries


def _build_zone_definition_3x3(frame_width: int, frame_height: int) -> dict[str, Any]:
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))
    x_edges = (0, width // 3, (2 * width) // 3, width)
    y_edges = (0, height // 3, (2 * height) // 3, height)

    zones: dict[str, dict[str, int]] = {}
    labels: list[str] = []
    for row_idx in range(3):
        for col_idx in range(3):
            label = _ZONE_LABELS_3X3[(row_idx * 3) + col_idx]
            labels.append(label)
            zones[label] = {
                "x1": int(x_edges[col_idx]),
                "y1": int(y_edges[row_idx]),
                "x2": int(x_edges[col_idx + 1]),
                "y2": int(y_edges[row_idx + 1]),
            }
    return {
        "layout": "3x3",
        "frame_width": width,
        "frame_height": height,
        "labels": labels,
        "zones": zones,
    }


def _zone_label_for_row(
    *,
    row: CanonicalTrackRow,
    frame_width: int,
    frame_height: int,
) -> str:
    center_x, center_y = _row_center(row)
    width = max(1.0, float(frame_width))
    height = max(1.0, float(frame_height))
    col_idx = max(0, min(2, int((center_x / width) * 3)))
    row_idx = max(0, min(2, int((center_y / height) * 3)))
    return _ZONE_LABELS_3X3[(row_idx * 3) + col_idx]


def _entity_type_for_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in _PERSON_LABELS:
        return "person"
    return "object"


def _build_simplified_video_entities(
    *,
    canonical_rows: list[CanonicalTrackRow],
    class_name_map: dict[int, str],
    frame_width: int,
    frame_height: int,
    sample_fps: int,
    max_entities: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[CanonicalTrackRow]] = {}
    for row in canonical_rows:
        grouped.setdefault(int(row.global_id), []).append(row)

    ranked_entities: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for global_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(rows, key=lambda item: int(item.t_ms))
        class_id = int(ordered[0].class_id)
        label = class_name_map.get(class_id, f"class_{class_id}")
        entity_type = _entity_type_for_label(label)

        appearance_ranges = [
            {
                "start_ms": int(span_rows[0].t_ms),
                "end_ms": int(span_rows[-1].t_ms),
            }
            for span_rows in _split_track_spans(rows=ordered, sample_fps=sample_fps)
            if span_rows
        ]

        zones_visited: list[str] = []
        seen_zones: set[str] = set()
        zone_occupancy: dict[str, int] = {}
        zone_transitions: list[dict[str, Any]] = []
        last_zone: str | None = None

        for row in ordered:
            zone = _zone_label_for_row(
                row=row,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            zone_occupancy[zone] = zone_occupancy.get(zone, 0) + 1
            if zone not in seen_zones:
                zones_visited.append(zone)
                seen_zones.add(zone)
            if last_zone is not None and zone != last_zone:
                zone_transitions.append(
                    {
                        "from": last_zone,
                        "to": zone,
                        "at_ms": int(row.t_ms),
                    }
                )
            last_zone = zone

        first_seen_ms = int(ordered[0].t_ms)
        last_seen_ms = int(ordered[-1].t_ms)
        entity = {
            "entity_id": f"{entity_type}-{global_id}",
            "global_track_id": int(global_id),
            "entity_type": entity_type,
            "label": label,
            "first_seen_ms": first_seen_ms,
            "last_seen_ms": last_seen_ms,
            "appearance_ranges_ms": appearance_ranges,
            "zones_visited": zones_visited,
            "zone_occupancy": {
                key: zone_occupancy[key] for key in sorted(zone_occupancy)
            },
            "zone_transitions": zone_transitions,
            "evidence_timestamps_ms": [
                first_seen_ms,
                last_seen_ms,
            ][: _DEFAULT_EVIDENCE_POINTS],
        }
        rank_key = (
            -len(ordered),
            -(last_seen_ms - first_seen_ms),
            int(global_id),
        )
        ranked_entities.append((rank_key, entity))

    ranked_entities.sort(key=lambda item: item[0])
    return [entity for _, entity in ranked_entities[: max(1, max_entities)]]


def _build_simplified_video_summary(
    *,
    canonical_rows: list[CanonicalTrackRow],
    class_name_map: dict[int, str],
    frame_width: int,
    frame_height: int,
    sample_fps: int,
    max_entities: int,
    zone_definition_strategy: ZoneDefinitionStrategy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    zone_definition = zone_definition_strategy(frame_width, frame_height)
    entities = _build_simplified_video_entities(
        canonical_rows=canonical_rows,
        class_name_map=class_name_map,
        frame_width=frame_width,
        frame_height=frame_height,
        sample_fps=sample_fps,
        max_entities=max_entities,
    )
    return zone_definition, entities


def _write_compact_json(*, payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)


def _empty_result_payload(
    *,
    enabled: bool,
    output_mode: str,
    embedding_mode: str | None = None,
    backend_strategy: str = _DEFAULT_BACKEND_STRATEGY,
    stitch_strategy: str = _DEFAULT_STITCH_STRATEGY,
    zone_strategy: str = _DEFAULT_ZONE_STRATEGY,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": enabled,
        "method": _DEFAULT_METHOD,
        "embedding_mode": embedding_mode,
        "output_mode": output_mode,
        "backend_strategy": backend_strategy,
        "stitch_strategy": stitch_strategy,
        "zone_strategy": zone_strategy,
        "tracks": [],
        "scenes": [],
        "zone_definition": None,
        "entities": [],
        "stats": {
            "chunk_count": 0,
            "row_count": 0,
            "track_count": 0,
            "scene_count": 0,
            "span_count": 0,
            "key_box_count": 0,
        },
        "artifacts": {},
    }
    if error:
        payload["error"] = error
    return payload


def _tracking_backend_strategy_registry() -> dict[str, Callable[..., Any]]:
    return {
        _DEFAULT_BACKEND_STRATEGY: _extract_chunk_local_tracks,
    }


def _stitch_strategy_registry() -> dict[str, Callable[..., Any]]:
    return {
        _DEFAULT_STITCH_STRATEGY: _stitch_boundary,
    }


def _zone_definition_strategy_registry() -> dict[str, Callable[..., Any]]:
    return {
        _DEFAULT_ZONE_STRATEGY: _build_zone_definition_3x3,
    }


def run_parallel_chunked_tracking(
    *,
    video_path: str,
    settings: Any,
    scenes: list[tuple[float, float]] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run chunked tracking and emit deterministic video-level summaries."""
    config = ChunkedTrackingConfig.from_settings(settings)
    output_mode = config.normalized_output_mode()
    ground_truth_backend = config.normalized_ground_truth_backend()
    backend_strategy_id = config.normalized_backend_strategy()
    stitch_strategy_id = config.normalized_stitch_strategy()
    zone_strategy_id = config.normalized_zone_strategy()

    if not config.enabled:
        return _empty_result_payload(
            enabled=False,
            output_mode=output_mode,
            embedding_mode=None,
            backend_strategy=backend_strategy_id,
            stitch_strategy=stitch_strategy_id,
            zone_strategy=zone_strategy_id,
        )

    chunk_extraction_strategy = _resolve_strategy(
        strategy_kind="tracking_backend",
        strategy_id=backend_strategy_id,
        registry=_tracking_backend_strategy_registry(),
    )
    stitch_strategy = _resolve_strategy(
        strategy_kind="stitch",
        strategy_id=stitch_strategy_id,
        registry=_stitch_strategy_registry(),
    )
    zone_definition_strategy = _resolve_strategy(
        strategy_kind="zone",
        strategy_id=zone_strategy_id,
        registry=_zone_definition_strategy_registry(),
    )

    embedder, embedding_mode = _build_embedder(config)

    cap = cv2.VideoCapture(video_path)
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if native_fps <= 0.0:
        native_fps = 25.0
    frame_width = max(1, frame_width)
    frame_height = max(1, frame_height)
    duration_sec = (float(frame_count) / native_fps) if frame_count > 0 else 0.0
    chunks = _plan_chunks(duration_sec, config.chunk_duration_sec, config.overlap_sec)

    zone_definition = zone_definition_strategy(frame_width, frame_height)
    if not chunks:
        payload = _empty_result_payload(
            enabled=True,
            output_mode=output_mode,
            embedding_mode=embedding_mode,
            backend_strategy=backend_strategy_id,
            stitch_strategy=stitch_strategy_id,
            zone_strategy=zone_strategy_id,
        )
        if output_mode != "legacy":
            payload["zone_definition"] = zone_definition
        return payload

    deduped_rows: dict[
        tuple[int, int, int, tuple[int, int, int, int]],
        tuple[int, CanonicalTrackRow],
    ] = {}
    active_global_tracks: dict[str, _GlobalTrackState] = {}
    next_global_index = 1
    class_name_map: dict[int, str] = {}

    extracted_by_chunk_id: dict[int, ChunkExtractionResult] = {}
    chunk_worker_limit = max(1, min(len(chunks), int(config.chunk_max_workers)))
    if chunk_worker_limit == 1:
        for chunk in chunks:
            extracted_by_chunk_id[chunk.chunk_id] = chunk_extraction_strategy(
                video_path=video_path,
                chunk=chunk,
                config=config,
                native_fps=native_fps,
                embedder=embedder,
            )
    else:
        with ThreadPoolExecutor(max_workers=chunk_worker_limit) as pool:
            future_by_chunk_id = {
                pool.submit(
                    chunk_extraction_strategy,
                    video_path=video_path,
                    chunk=chunk,
                    config=config,
                    native_fps=native_fps,
                    embedder=embedder,
                ): chunk.chunk_id
                for chunk in chunks
            }
            for future in as_completed(future_by_chunk_id):
                chunk_id = future_by_chunk_id[future]
                extracted_by_chunk_id[chunk_id] = future.result()

    for chunk in sorted(chunks, key=lambda item: int(item.chunk_id)):
        extracted = extracted_by_chunk_id.get(chunk.chunk_id)
        if extracted is None:
            continue
        if len(extracted) == 3:
            chunk_rows, local_tracks, chunk_class_name_map = extracted
        else:  # pragma: no cover - backward compatibility for monkeypatched tests
            chunk_rows, local_tracks = extracted  # type: ignore[misc]
            chunk_class_name_map = {}  # type: ignore[assignment]
        for class_id, class_name in chunk_class_name_map.items():
            class_name_map[int(class_id)] = str(class_name)
        head_signatures = _build_track_signatures(
            local_tracks,
            window_start_sec=chunk.start_sec,
            window_end_sec=min(chunk.end_sec, chunk.start_sec + config.overlap_sec),
            velocity_window=config.velocity_window,
        )
        mapping, next_global_index = stitch_strategy(
            active_global_tracks=active_global_tracks,
            local_head_signatures=head_signatures,
            min_cosine=config.min_cosine,
            min_iou=config.min_iou,
            overlap_sec=config.overlap_sec,
            next_global_index=next_global_index,
        )
        for local_id in sorted(local_tracks):
            if local_id in mapping:
                continue
            mapping[local_id] = f"global_{next_global_index}"
            next_global_index += 1

        for row in chunk_rows:
            global_id_raw = mapping.get(str(row["local_id"]))
            if not global_id_raw:
                continue
            bbox = [int(value) for value in row["bbox_xyxy"]]
            global_id = _global_id_as_int(global_id_raw)
            payload_row = _normalize_canonical_row(
                t_ms=int(round(float(row["t_sec"]) * 1000.0)),
                global_id=global_id,
                class_id=int(row["class_id"]),
                conf=round(float(row["conf"]), 6),
                x1=bbox[0],
                y1=bbox[1],
                x2=bbox[2],
                y2=bbox[3],
            )
            dedupe_key = (
                int(row["frame_idx"]),
                global_id,
                int(payload_row.class_id),
                (int(payload_row.x1), int(payload_row.y1), int(payload_row.x2), int(payload_row.y2)),
            )
            previous = deduped_rows.get(dedupe_key)
            if previous is None or chunk.chunk_id >= previous[0]:
                deduped_rows[dedupe_key] = (chunk.chunk_id, payload_row)

        tail_signatures = _build_track_signatures(
            local_tracks,
            window_start_sec=max(chunk.start_sec, chunk.end_sec - config.overlap_sec),
            window_end_sec=chunk.end_sec,
            velocity_window=config.velocity_window,
        )
        next_active_tracks: dict[str, _GlobalTrackState] = {}
        for local_id, signature in tail_signatures.items():
            global_id = mapping.get(local_id)
            if not global_id:
                continue
            previous = active_global_tracks.get(global_id)
            merged_embedding = _mean_embedding(
                [
                    previous.embedding if previous is not None else None,
                    signature.embedding,
                ]
            )
            next_active_tracks[global_id] = _GlobalTrackState(
                global_id=global_id,
                class_id=signature.class_id,
                embedding=merged_embedding,
                bbox_end=list(signature.bbox_end),
                velocity_xy=signature.velocity_xy,
                last_t=signature.end_t,
            )
        active_global_tracks = next_active_tracks

    canonical_rows = sorted(
        [value[1] for value in deduped_rows.values()],
        key=lambda item: (
            int(item.global_id),
            int(item.t_ms),
            int(item.class_id),
        ),
    )

    output_root = _resolve_output_dir(video_path=video_path, output_dir=output_dir)
    persisted_rows, artifacts = _persist_and_reload_canonical_rows(
        rows=canonical_rows,
        backend=ground_truth_backend,
        output_root=output_root,
    )
    zone_definition, simplified_entities = _build_simplified_video_summary(
        canonical_rows=persisted_rows,
        class_name_map=class_name_map,
        frame_width=frame_width,
        frame_height=frame_height,
        sample_fps=config.sample_fps,
        max_entities=config.max_entities,
        zone_definition_strategy=zone_definition_strategy,
    )

    compact_tracks: list[dict[str, Any]] = []
    legacy_scene_packets: list[dict[str, Any]] = []
    if output_mode in {"legacy", "dual"}:
        compact_tracks = _build_compact_tracks(
            canonical_rows=persisted_rows,
            sample_fps=config.sample_fps,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        legacy_scene_packets = _build_scene_packets(
            tracks=compact_tracks,
            scenes=scenes or [],
        )
        legacy_payload = {
            "enabled": True,
            "method": _DEFAULT_METHOD,
            "tracks": compact_tracks,
            "scenes": legacy_scene_packets,
        }
        legacy_path = output_root / "tracks.compact.json"
        _write_compact_json(payload=legacy_payload, output_path=legacy_path)
        artifacts["tracks_compact_json"] = str(legacy_path)

    if output_mode in {"summary_v2", "dual"}:
        summary_payload = {
            "enabled": True,
            "method": _DEFAULT_METHOD,
            "zone_definition": zone_definition,
            "entities": simplified_entities,
        }
        summary_path = output_root / "tracks.video_summary.json"
        _write_compact_json(payload=summary_payload, output_path=summary_path)
        artifacts["video_summary_json"] = str(summary_path)

    selected_tracks: list[dict[str, Any]]
    selected_scenes: list[dict[str, Any]]
    selected_zone_definition: dict[str, Any] | None
    selected_entities: list[dict[str, Any]]
    if output_mode == "legacy":
        selected_tracks = compact_tracks
        selected_scenes = legacy_scene_packets
        selected_zone_definition = None
        selected_entities = []
    else:
        selected_tracks = []
        selected_scenes = []
        selected_zone_definition = zone_definition
        selected_entities = simplified_entities

    span_count = sum(len(track.get("spans", [])) for track in compact_tracks)
    key_box_count = sum(
        len(span.get("k", []))
        for track in compact_tracks
        for span in track.get("spans", [])
    )
    payload = {
        "enabled": True,
        "method": _DEFAULT_METHOD,
        "embedding_mode": embedding_mode,
        "output_mode": output_mode,
        "backend_strategy": backend_strategy_id,
        "stitch_strategy": stitch_strategy_id,
        "zone_strategy": zone_strategy_id,
        "tracks": selected_tracks,
        "scenes": selected_scenes,
        "zone_definition": selected_zone_definition,
        "entities": selected_entities,
        "stats": {
            "chunk_count": len(chunks),
            "row_count": len(persisted_rows),
            "track_count": len(selected_tracks),
            "scene_count": len(selected_scenes),
            "span_count": span_count,
            "key_box_count": key_box_count,
            "entity_count": len(selected_entities),
            "zone_count": len(zone_definition.get("zones", {})),
            "chunk_worker_limit": chunk_worker_limit,
            "legacy_track_count": len(compact_tracks),
            "legacy_scene_count": len(legacy_scene_packets),
            "scene_entity_count": len(simplified_entities),
        },
        "artifacts": artifacts,
    }
    if output_mode == "dual":
        payload["rollout"] = {
            "mode": "dual",
            "primary_output": "summary_v2",
            "rollback_hint": "Set PARALLEL_TRACKING_OUTPUT_MODE=legacy to revert output shape.",
        }
    return payload
