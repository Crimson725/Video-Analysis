"""Scene detection and keyframe extraction."""

from pathlib import Path

import cv2
from scenedetect import ContentDetector, detect


def detect_scenes(video_path: str) -> list[tuple[float, float]]:
    """Detect scene boundaries using ContentDetector. Returns list of (start_time, end_time) in seconds."""
    scene_list = detect(video_path, ContentDetector())
    result: list[tuple[float, float]] = []
    for start_tc, end_tc in scene_list:
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        result.append((start_sec, end_sec))
    return result


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

        # Format timestamp as HH:MM:SS.mmm
        hours = int(mid_sec // 3600)
        mins = int((mid_sec % 3600) // 60)
        secs = mid_sec % 60
        timestamp = f"{hours:02d}:{mins:02d}:{secs:06.3f}"

        frames.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image": image,
            }
        )

    cap.release()
    return frames


def save_original_frames(
    frames: list[dict], job_id: str, static_dir: str
) -> None:
    """Save each frame's image to static/{job_id}/original/frame_{N}.jpg"""
    base = Path(static_dir) / job_id / "original"
    base.mkdir(parents=True, exist_ok=True)
    for f in frames:
        path = base / f"frame_{f['frame_id']}.jpg"
        cv2.imwrite(str(path), f["image"])
