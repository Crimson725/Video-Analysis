"""Scene detection and keyframe extraction."""

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from scenedetect import ContentDetector, detect

if TYPE_CHECKING:
    from app.storage import MediaStore


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


def extract_keyframes(
    video_path: str, scenes: list[tuple[float, float]]
) -> list[dict]:
    """
    Extract the middle keyframe of each scene.
    Returns list of dicts with frame_id, timestamp, image (numpy array BGR).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[dict] = []

    for frame_id, (start_sec, end_sec) in enumerate(scenes):
        mid_sec = (start_sec + end_sec) / 2
        mid_frame_idx = int(mid_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, image = cap.read()
        if not ret:
            continue

        timestamp = _format_timestamp(mid_sec)

        frames.append(
            {
                "frame_id": frame_id,
                "scene_id": frame_id,
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
                raise RuntimeError(f"Failed to encode original frame {frame_id} as JPEG")
            media_store.upload_frame_image(
                job_id=job_id,
                frame_kind="original",
                frame_id=frame_id,
                image_bytes=encoded.tobytes(),
            )
