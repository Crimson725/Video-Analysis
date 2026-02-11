"""Frame analysis pipeline: segmentation, detection, face recognition."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image
from pydantic import ValidationError

from app.schemas import FrameResult
from app.storage import (
    FrameKind,
    MediaStoreError,
    build_analysis_key,
    build_frame_key,
)
from app.toon import ToonConversionError, convert_json_to_toon

if TYPE_CHECKING:
    from app.storage import MediaStore

logger = logging.getLogger(__name__)


def _to_int_coords(coord: float) -> int:
    """Round coordinate to integer for JSON output."""
    return int(round(coord))


def _persist_visualization(
    image: np.ndarray,
    local_path: Path | None,
    media_store: "MediaStore | None",
    job_id: str,
    frame_kind: FrameKind,
    frame_id: int,
) -> None:
    """Persist visualization locally and optionally upload to object storage."""
    if local_path is not None:
        cv2.imwrite(str(local_path), image)

    if media_store is not None:
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError(f"Failed to encode {frame_kind} frame {frame_id} as JPEG")
        media_store.upload_frame_image(
            job_id=job_id,
            frame_kind=frame_kind,
            frame_id=frame_id,
            image_bytes=encoded.tobytes(),
        )


def run_segmentation(
    image: np.ndarray,
    model: Any,
    job_id: str,
    frame_id: int,
    local_dir: str | None,
    media_store: "MediaStore | None" = None,
) -> list[dict]:
    """Run YOLO segmentation, persist visualization, and return structured data."""
    results = model(image, verbose=False)
    result = results[0]

    plot_img = result.plot()
    local_path: Path | None = None
    if local_dir:
        base = Path(local_dir) / job_id / "seg"
        base.mkdir(parents=True, exist_ok=True)
        local_path = base / f"frame_{frame_id}.jpg"
    _persist_visualization(plot_img, local_path, media_store, job_id, "seg", frame_id)

    # Extract structured data
    items: list[dict] = []
    if result.masks is not None and result.boxes is not None:
        names = result.names or {}
        for i, (mask_xy, cls_id) in enumerate(
            zip(result.masks.xy, result.boxes.cls.cpu().numpy())
        ):
            class_name = names.get(int(cls_id), str(int(cls_id)))
            polygon = [
                [_to_int_coords(x), _to_int_coords(y)] for x, y in mask_xy
            ]
            items.append(
                {
                    "object_id": i + 1,
                    "class": class_name,
                    "mask_polygon": polygon,
                }
            )
    return items


def run_detection(
    image: np.ndarray,
    model: Any,
    job_id: str,
    frame_id: int,
    local_dir: str | None,
    media_store: "MediaStore | None" = None,
) -> list[dict]:
    """Run YOLO detection, persist visualization, and return structured data."""
    results = model(image, verbose=False)
    result = results[0]

    plot_img = result.plot()
    local_path: Path | None = None
    if local_dir:
        base = Path(local_dir) / job_id / "det"
        base.mkdir(parents=True, exist_ok=True)
        local_path = base / f"frame_{frame_id}.jpg"
    _persist_visualization(plot_img, local_path, media_store, job_id, "det", frame_id)

    # Extract structured data
    items: list[dict] = []
    if result.boxes is not None:
        names = result.names or {}
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        for i in range(len(xyxy)):
            box = xyxy[i]
            label = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
            items.append(
                {
                    "label": label,
                    "confidence": float(conf[i]),
                    "box": [
                        _to_int_coords(box[0]),
                        _to_int_coords(box[1]),
                        _to_int_coords(box[2]),
                        _to_int_coords(box[3]),
                    ],
                }
            )
    return items


def run_face_recognition(
    image: np.ndarray,
    face_detector: Any,
    job_id: str,
    frame_id: int,
    local_dir: str | None,
    media_store: "MediaStore | None" = None,
    confidence_threshold: float = 0.9,
) -> list[dict]:
    """Run MTCNN face detection, persist visualization, and return structured data."""
    # Convert BGR (OpenCV) to RGB for MTCNN
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_array)

    # Detect faces — returns (boxes, probs) or (None, None)
    boxes, probs = face_detector.detect(pil_img)

    vis_img = image.copy()

    # Handle no detections
    if boxes is None or probs is None:
        local_path: Path | None = None
        if local_dir:
            base = Path(local_dir) / job_id / "face"
            base.mkdir(parents=True, exist_ok=True)
            local_path = base / f"frame_{frame_id}.jpg"
        _persist_visualization(vis_img, local_path, media_store, job_id, "face", frame_id)
        return []

    items: list[dict] = []
    face_id = 0
    for i in range(len(boxes)):
        prob = float(probs[i])
        if prob < confidence_threshold:
            continue

        x1 = _to_int_coords(boxes[i][0])
        y1 = _to_int_coords(boxes[i][1])
        x2 = _to_int_coords(boxes[i][2])
        y2 = _to_int_coords(boxes[i][3])

        # Draw bounding box on visualization
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face_id += 1
        items.append(
            {
                "face_id": face_id,
                "confidence": prob,
                "coordinates": [x1, y1, x2, y2],
            }
        )

    local_path: Path | None = None
    if local_dir:
        base = Path(local_dir) / job_id / "face"
        base.mkdir(parents=True, exist_ok=True)
        local_path = base / f"frame_{frame_id}.jpg"
    _persist_visualization(vis_img, local_path, media_store, job_id, "face", frame_id)

    return items


def _cleanup_analysis_artifacts(
    media_store: "MediaStore",
    job_id: str,
    frame_id: int,
) -> None:
    """Best-effort cleanup for per-frame analysis artifacts."""
    for artifact_kind in ("json", "toon"):
        object_key = build_analysis_key(job_id, artifact_kind, frame_id)
        try:
            media_store.delete_object(object_key)
        except MediaStoreError:
            logger.warning(
                "Best-effort cleanup failed for analysis artifact: %s", object_key
            )


def _persist_analysis_artifacts(
    media_store: "MediaStore",
    frame_payload: dict[str, Any],
    job_id: str,
    frame_id: int,
) -> None:
    """Validate payload, persist JSON artifact, convert to TOON, and persist TOON."""
    try:
        validated = FrameResult.model_validate(frame_payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"Frame payload contract validation failed for frame {frame_id}"
        ) from exc

    json_payload = validated.model_dump_json(by_alias=True).encode("utf-8")
    media_store.upload_analysis_artifact(job_id, "json", frame_id, json_payload)

    try:
        toon_payload = convert_json_to_toon(json_payload)
        media_store.upload_analysis_artifact(job_id, "toon", frame_id, toon_payload)
    except (MediaStoreError, ToonConversionError, RuntimeError) as exc:
        _cleanup_analysis_artifacts(media_store, job_id, frame_id)
        raise RuntimeError(
            f"Failed to persist analysis artifacts for frame {frame_id}"
        ) from exc


def analyze_frame(
    frame_data: dict,
    models: Any,
    job_id: str,
    local_dir: str | None,
    media_store: "MediaStore | None" = None,
) -> dict:
    """Run all three analysis tasks on a frame and return FrameResult-compatible dict."""
    image = frame_data["image"]
    frame_id = frame_data["frame_id"]
    timestamp = frame_data["timestamp"]

    seg_items = run_segmentation(
        image, models.segmenter, job_id, frame_id, local_dir, media_store
    )
    det_items = run_detection(
        image, models.detector, job_id, frame_id, local_dir, media_store
    )
    face_items = run_face_recognition(
        image, models.face_detector, job_id, frame_id, local_dir, media_store
    )

    if media_store is not None:
        files = {
            "original": build_frame_key(job_id, "original", frame_id),
            "segmentation": build_frame_key(job_id, "seg", frame_id),
            "detection": build_frame_key(job_id, "det", frame_id),
            "face": build_frame_key(job_id, "face", frame_id),
        }
    else:
        base_path = f"/static/{job_id}"
        files = {
            "original": f"{base_path}/original/frame_{frame_id}.jpg",
            "segmentation": f"{base_path}/seg/frame_{frame_id}.jpg",
            "detection": f"{base_path}/det/frame_{frame_id}.jpg",
            "face": f"{base_path}/face/frame_{frame_id}.jpg",
        }

    frame_payload: dict[str, Any] = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "files": files,
        "analysis": {
            "semantic_segmentation": seg_items,
            "object_detection": det_items,
            "face_recognition": face_items,
        },
        "analysis_artifacts": {
            "json": build_analysis_key(job_id, "json", frame_id),
            "toon": build_analysis_key(job_id, "toon", frame_id),
        },
    }

    if media_store is not None:
        _persist_analysis_artifacts(media_store, frame_payload, job_id, frame_id)

    return frame_payload
