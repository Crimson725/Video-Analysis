"""Test-only local CLI for frame analysis + chunked tracking."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import uuid

import cv2

from app import analysis, parallel_tracking, scene
from app.config import Settings
from app.models import ModelLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local test pipeline on one video: keyframe/frame analysis + "
            "chunked object tracking."
        )
    )
    parser.add_argument("video", help="Path to input video file")
    return parser.parse_args()


def _local_frame_files(base_dir: Path, job_id: str, frame_id: int) -> dict[str, str]:
    root = base_dir / job_id
    return {
        "original": str(root / "original" / f"frame_{frame_id}.jpg"),
        "segmentation": str(root / "seg" / f"frame_{frame_id}.jpg"),
        "detection": str(root / "det" / f"frame_{frame_id}.jpg"),
        "face": str(root / "face" / f"frame_{frame_id}.jpg"),
    }


def main() -> int:
    args = _parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    run_dir = (video_path.parent / f"video-analysis-test-{uuid.uuid4().hex[:8]}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    output_base = run_dir / "outputs"
    output_base.mkdir(parents=True, exist_ok=True)

    job_id = f"test_{uuid.uuid4().hex[:8]}"
    settings = replace(
        Settings.from_env(),
        enable_parallel_chunked_tracking_pipeline=True,
    )
    models = ModelLoader.get()

    scenes = scene.detect_scenes(str(video_path))
    if not scenes:
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        duration_sec = (frame_count / fps) if fps > 0 else 0.0
        scenes = [(0.0, max(0.0, duration_sec))]
    keyframes = scene.extract_keyframes(str(video_path), scenes)
    scene.save_original_frames(
        keyframes,
        job_id=job_id,
        local_dir=str(output_base),
        media_store=None,
    )

    face_tracker = analysis.FaceIdentityTracker()
    object_tracker = analysis.ObjectTrackTracker()
    frame_results: list[dict] = []

    analysis_dir = output_base / job_id / "analysis" / "json"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for frame in keyframes:
        frame_result = analysis.analyze_frame(
            frame_data=frame,
            models=models,
            job_id=job_id,
            local_dir=str(output_base),
            media_store=None,
            face_tracker=face_tracker,
            object_tracker=object_tracker,
        )
        frame_id = int(frame_result["frame_id"])
        frame_result["files"] = _local_frame_files(output_base, job_id, frame_id)

        analysis_json_path = analysis_dir / f"frame_{frame_id}.json"
        with analysis_json_path.open("w", encoding="utf-8") as handle:
            json.dump(frame_result["analysis"], handle, indent=2)
        frame_result["analysis_artifacts"]["json"] = str(analysis_json_path)
        frame_results.append(frame_result)

    tracking_output_dir = output_base / job_id / "tracking"
    chunked_payload = parallel_tracking.run_parallel_chunked_tracking(
        video_path=str(video_path),
        settings=settings,
        scenes=scenes,
        output_dir=tracking_output_dir,
    )

    result_payload = {
        "job_id": job_id,
        "video_path": str(video_path),
        "run_dir": str(run_dir),
        "frames": frame_results,
    }
    result_path = output_base / job_id / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Result JSON: {result_path}")
    output_mode = str(chunked_payload.get("output_mode", "summary_v2"))
    if output_mode == "legacy":
        artifact_name = "tracks.compact.json"
    else:
        artifact_name = "tracks.video_summary.json"
    print(f"Tracking summary JSON: {tracking_output_dir / artifact_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
