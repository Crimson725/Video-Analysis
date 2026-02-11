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
