"""Frame analysis pipeline: segmentation, detection, face recognition."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _to_int_coords(coord: float) -> int:
    """Round coordinate to integer for JSON output."""
    return int(round(coord))


def run_segmentation(
    image: np.ndarray,
    model: Any,
    job_id: str,
    frame_id: int,
    static_dir: str,
) -> list[dict]:
    """Run YOLO segmentation, save visualization, return structured data."""
    results = model(image, verbose=False)
    result = results[0]

    # Save visualization
    base = Path(static_dir) / job_id / "seg"
    base.mkdir(parents=True, exist_ok=True)
    plot_img = result.plot()
    cv2.imwrite(str(base / f"frame_{frame_id}.jpg"), plot_img)

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
    static_dir: str,
) -> list[dict]:
    """Run YOLO detection, save visualization, return structured data."""
    results = model(image, verbose=False)
    result = results[0]

    # Save visualization
    base = Path(static_dir) / job_id / "det"
    base.mkdir(parents=True, exist_ok=True)
    plot_img = result.plot()
    cv2.imwrite(str(base / f"frame_{frame_id}.jpg"), plot_img)

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
    static_dir: str,
    confidence_threshold: float = 0.9,
) -> list[dict]:
    """Run MTCNN face detection, save visualization, return structured data."""
    # Convert BGR (OpenCV) to RGB for MTCNN
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_array)

    # Detect faces — returns (boxes, probs) or (None, None)
    boxes, probs = face_detector.detect(pil_img)

    vis_img = image.copy()

    # Handle no detections
    if boxes is None or probs is None:
        base = Path(static_dir) / job_id / "face"
        base.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(base / f"frame_{frame_id}.jpg"), vis_img)
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

    # Save visualization
    base = Path(static_dir) / job_id / "face"
    base.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(base / f"frame_{frame_id}.jpg"), vis_img)

    return items


def analyze_frame(
    frame_data: dict,
    models: Any,
    job_id: str,
    static_dir: str,
) -> dict:
    """Run all three analysis tasks on a frame and return FrameResult-compatible dict."""
    image = frame_data["image"]
    frame_id = frame_data["frame_id"]
    timestamp = frame_data["timestamp"]

    seg_items = run_segmentation(
        image, models.segmenter, job_id, frame_id, static_dir
    )
    det_items = run_detection(
        image, models.detector, job_id, frame_id, static_dir
    )
    face_items = run_face_recognition(
        image, models.face_detector, job_id, frame_id, static_dir
    )

    base_path = f"/static/{job_id}"
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "files": {
            "original": f"{base_path}/original/frame_{frame_id}.jpg",
            "segmentation": f"{base_path}/seg/frame_{frame_id}.jpg",
            "detection": f"{base_path}/det/frame_{frame_id}.jpg",
            "face": f"{base_path}/face/frame_{frame_id}.jpg",
        },
        "analysis": {
            "semantic_segmentation": seg_items,
            "object_detection": det_items,
            "face_recognition": face_items,
        },
    }
