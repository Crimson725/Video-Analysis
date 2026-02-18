"""Scene detection and keyframe extraction."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

if TYPE_CHECKING:
    from app.storage import MediaStore

_DEFAULT_KEYFRAME_SCAN_FPS = 2.0
_DEFAULT_KEYFRAME_NMS_DELTA_SEC = 0.7
_KEYFRAME_CHANGE_SCORE_FLOOR = 0.03
_DOWNSCALE_SIZE = (160, 90)


@dataclass(frozen=True)
class _ScoredCandidate:
    timestamp_sec: float
    frame_index: int
    score: float


def detect_scenes(video_path: str) -> list[tuple[float, float]]:
    """Detect scene boundaries using ContentDetector. Returns list of (start_time, end_time) in seconds."""
    scene_list = detect(video_path, ContentDetector())
    result: list[tuple[float, float]] = []
    for start_tc, end_tc in scene_list:
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        result.append((start_sec, end_sec))
    return result


def _format_timestamp(seconds_total: float) -> str:
    """Format floating-point seconds as HH:MM:SS.mmm."""
    hours = int(seconds_total // 3600)
    mins = int((seconds_total % 3600) // 60)
    secs = seconds_total % 60
    return f"{hours:02d}:{mins:02d}:{secs:06.3f}"


def _default_scene_budget(duration_sec: float) -> int:
    """Select default frame budget from scene duration."""
    if duration_sec <= 10.0:
        return 12
    if duration_sec <= 60.0:
        return 30
    return 60


def _safe_frame_index(timestamp_sec: float, fps: float) -> int:
    return max(0, int(max(0.0, timestamp_sec) * fps))


def _clamp_scene_end_anchor(start_sec: float, end_sec: float, fps: float) -> float:
    if fps <= 0.0:
        return float(start_sec)
    return max(float(start_sec), float(end_sec) - (1.0 / fps))


def _build_scan_timestamps(
    start_sec: float,
    end_sec: float,
    scan_fps: float,
    end_anchor_sec: float,
) -> list[float]:
    """Build scan timestamps ts + i/r, then clamp to in-range decodable end."""
    if end_sec < start_sec:
        return []

    effective_scan_fps = (
        float(scan_fps) if float(scan_fps) > 0 else _DEFAULT_KEYFRAME_SCAN_FPS
    )
    sample_count = int(math.floor((end_sec - start_sec) * effective_scan_fps))
    raw = [start_sec + (i / effective_scan_fps) for i in range(sample_count + 1)]
    if not raw:
        raw = [start_sec]

    deduped: list[float] = []
    last: float | None = None
    for ts in raw:
        bounded = min(max(ts, start_sec), end_anchor_sec)
        if last is None or abs(bounded - last) > 1e-6:
            deduped.append(bounded)
            last = bounded
    return deduped


def _downscale_grayscale(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, _DOWNSCALE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _robust_change_threshold(scores: list[float]) -> float:
    if not scores:
        return _KEYFRAME_CHANGE_SCORE_FLOOR
    score_array = np.asarray(scores, dtype=np.float32)
    median = float(np.median(score_array))
    mad = float(np.median(np.abs(score_array - median)))
    return max(median + (2.0 * mad), _KEYFRAME_CHANGE_SCORE_FLOOR)


def _temporal_nms(
    candidates: list[_ScoredCandidate],
    delta_sec: float,
) -> list[_ScoredCandidate]:
    """Suppress near-duplicate candidates in time, retaining stronger peaks."""
    if not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda item: (item.timestamp_sec, item.frame_index),
    )
    kept: list[_ScoredCandidate] = []
    for candidate in ordered:
        if not kept:
            kept.append(candidate)
            continue

        previous = kept[-1]
        if candidate.timestamp_sec - previous.timestamp_sec <= delta_sec:
            if candidate.score > previous.score:
                kept[-1] = candidate
            continue
        kept.append(candidate)

    return kept


def _collect_change_candidates(
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    fps: float,
    scan_fps: float,
) -> list[_ScoredCandidate]:
    """Decode low-rate scan and return score-filtered change candidates."""
    end_anchor_sec = _clamp_scene_end_anchor(start_sec, end_sec, fps)
    scan_times = _build_scan_timestamps(start_sec, end_sec, scan_fps, end_anchor_sec)
    if not scan_times:
        return []

    scored: list[_ScoredCandidate] = []
    previous_small: np.ndarray | None = None

    for timestamp_sec in scan_times:
        frame_index = _safe_frame_index(timestamp_sec, fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, image = cap.read()
        if not ret:
            continue

        small_gray = _downscale_grayscale(image)
        if previous_small is not None:
            score = float(np.mean(np.abs(small_gray - previous_small)))
            scored.append(
                _ScoredCandidate(
                    timestamp_sec=timestamp_sec,
                    frame_index=frame_index,
                    score=score,
                )
            )
        previous_small = small_gray

    threshold = _robust_change_threshold([item.score for item in scored])
    return [item for item in scored if item.score > threshold]


def _select_scene_extraction_points(
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> list[tuple[float, int]]:
    """Select per-scene timestamps/frame indices using anchors + change candidates."""
    if end_sec < start_sec:
        return []

    end_anchor_sec = _clamp_scene_end_anchor(start_sec, end_sec, fps)
    anchor_times = [float(start_sec), (start_sec + end_sec) / 2.0, end_anchor_sec]

    anchors: list[tuple[float, int]] = []
    anchor_frame_indices: set[int] = set()
    for timestamp_sec in sorted(anchor_times):
        frame_index = _safe_frame_index(timestamp_sec, fps)
        if frame_index in anchor_frame_indices:
            continue
        anchor_frame_indices.add(frame_index)
        anchors.append((timestamp_sec, frame_index))

    candidates = _collect_change_candidates(
        cap=cap,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        scan_fps=_DEFAULT_KEYFRAME_SCAN_FPS,
    )
    candidates = _temporal_nms(candidates, _DEFAULT_KEYFRAME_NMS_DELTA_SEC)
    candidates = [
        item for item in candidates if item.frame_index not in anchor_frame_indices
    ]

    remaining_slots = max(0, _default_scene_budget(end_sec - start_sec) - len(anchors))
    if len(candidates) > remaining_slots:
        candidates = sorted(
            candidates,
            key=lambda item: (-item.score, item.timestamp_sec, item.frame_index),
        )[:remaining_slots]

    # Dedupe by frame index to avoid decoding the same frame repeatedly.
    selected: dict[int, tuple[float, float]] = {
        frame_index: (timestamp_sec, float("inf"))
        for timestamp_sec, frame_index in anchors
    }
    for candidate in candidates:
        existing = selected.get(candidate.frame_index)
        if existing is None:
            selected[candidate.frame_index] = (candidate.timestamp_sec, candidate.score)
            continue
        existing_timestamp, existing_score = existing
        if candidate.score > existing_score or (
            candidate.score == existing_score
            and candidate.timestamp_sec < existing_timestamp
        ):
            selected[candidate.frame_index] = (candidate.timestamp_sec, candidate.score)

    return sorted(
        ((timestamp_sec, frame_index) for frame_index, (timestamp_sec, _) in selected.items()),
        key=lambda item: (item[0], item[1]),
    )


def extract_keyframes(video_path: str, scenes: list[tuple[float, float]]) -> list[dict]:
    """
    Extract robust keyframes per scene under a duration-tiered budget.
    Returns list of dicts with frame_id, scene_id, timestamp, image (numpy array BGR).
    """
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    extraction_plan: list[tuple[float, int, int]] = []
    for scene_id, (start_sec, end_sec) in enumerate(scenes):
        points = _select_scene_extraction_points(
            cap=cap,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            fps=fps,
        )
        for timestamp_sec, frame_index in points:
            extraction_plan.append((timestamp_sec, scene_id, frame_index))

    extraction_plan.sort(key=lambda item: (item[0], item[1], item[2]))

    frames: list[dict] = []
    for timestamp_sec, scene_id, frame_index in extraction_plan:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, image = cap.read()
        if not ret:
            continue

        timestamp = _format_timestamp(timestamp_sec)

        frames.append(
            {
                "frame_id": len(frames),
                "scene_id": scene_id,
                "timestamp": timestamp,
                "image": image,
            }
        )

    cap.release()
    return frames


def extract_tracking_frames(
    video_path: str,
    scenes: list[tuple[float, float]],
    *,
    sample_fps: int,
    max_samples_per_scene: int,
) -> list[dict]:
    """Extract continuous sampled frames for identity tracking per scene."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sampling_hz = max(1, int(sample_fps))
    max_samples = max(1, int(max_samples_per_scene))
    step_sec = 1.0 / float(sampling_hz)
    sampled_frames: list[dict] = []

    for scene_id, (start_sec, end_sec) in enumerate(scenes):
        if end_sec < start_sec:
            continue
        sample_times: list[float] = []
        cursor = float(start_sec)
        while cursor <= float(end_sec) and len(sample_times) < max_samples:
            sample_times.append(cursor)
            cursor += step_sec
        if not sample_times:
            sample_times = [float(start_sec)]

        for sample_index, second_mark in enumerate(sample_times):
            source_frame_idx = int(second_mark * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
            ret, image = cap.read()
            if not ret:
                continue

            # Keep deterministic monotonic IDs per scene.
            frame_id = scene_id * 1_000_000 + sample_index
            sampled_frames.append(
                {
                    "frame_id": frame_id,
                    "scene_id": scene_id,
                    "sample_index": sample_index,
                    "timestamp": _format_timestamp(second_mark),
                    "image": image,
                    "source_frame_index": source_frame_idx,
                    "is_tracking_frame": True,
                }
            )

    cap.release()
    return sampled_frames


def save_original_frames(
    frames: list[dict],
    job_id: str,
    local_dir: str,
    media_store: "MediaStore | None" = None,
) -> None:
    """Save local original frames and optionally upload them to object storage."""
    base = Path(local_dir) / job_id / "original"
    base.mkdir(parents=True, exist_ok=True)
    for f in frames:
        frame_id = int(f["frame_id"])
        image = f["image"]
        path = base / f"frame_{frame_id}.jpg"
        cv2.imwrite(str(path), image)

        if media_store is not None:
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise RuntimeError(
                    f"Failed to encode original frame {frame_id} as JPEG"
                )
            media_store.upload_frame_image(
                job_id=job_id,
                frame_kind="original",
                frame_id=frame_id,
                image_bytes=encoded.tobytes(),
            )
