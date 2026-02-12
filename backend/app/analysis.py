"""Frame analysis pipeline: segmentation, detection, face recognition."""

from dataclasses import dataclass
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


@dataclass
class _FaceTrack:
    """Internal state for a tracked anonymous face identity."""

    identity_num: int
    box: tuple[int, int, int, int]
    last_frame_id: int


class FaceIdentityTracker:
    """Assign stable anonymous identity IDs to faces across frames."""

    def __init__(self, iou_threshold: float = 0.35, max_frame_gap: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_frame_gap = max_frame_gap
        self._next_identity_num = 1
        self._tracks: dict[int, _FaceTrack] = {}

    @staticmethod
    def _intersection_over_union(
        box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    def _drop_stale_tracks(self, frame_id: int) -> None:
        stale = [
            identity_num
            for identity_num, track in self._tracks.items()
            if frame_id - track.last_frame_id > self.max_frame_gap
        ]
        for identity_num in stale:
            del self._tracks[identity_num]

    def assign_identity(
        self,
        box: tuple[int, int, int, int],
        frame_id: int,
        used_identities: set[int],
    ) -> int:
        """Return a stable identity number for the current face box."""
        self._drop_stale_tracks(frame_id)

        best_identity: int | None = None
        best_iou = 0.0

        for identity_num, track in self._tracks.items():
            if identity_num in used_identities:
                continue
            iou = self._intersection_over_union(box, track.box)
            if iou > best_iou:
                best_iou = iou
                best_identity = identity_num

        if best_identity is not None and best_iou >= self.iou_threshold:
            matched = self._tracks[best_identity]
            matched.box = box
            matched.last_frame_id = frame_id
            return best_identity

        identity_num = self._next_identity_num
        self._next_identity_num += 1
        self._tracks[identity_num] = _FaceTrack(
            identity_num=identity_num,
            box=box,
            last_frame_id=frame_id,
        )
        return identity_num


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


def _build_local_visualization_path(
    local_dir: str | None,
    job_id: str,
    frame_kind: FrameKind,
    frame_id: int,
) -> Path | None:
    """Build and ensure local path for per-frame visualization output."""
    if not local_dir:
        return None
    base = Path(local_dir) / job_id / frame_kind
    base.mkdir(parents=True, exist_ok=True)
    return base / f"frame_{frame_id}.jpg"


def _build_frame_files(
    job_id: str,
    frame_id: int,
    media_store: "MediaStore | None",
) -> dict[str, str]:
    """Build deterministic frame file references for local or object storage modes."""
    if media_store is not None:
        return {
            "original": build_frame_key(job_id, "original", frame_id),
            "segmentation": build_frame_key(job_id, "seg", frame_id),
            "detection": build_frame_key(job_id, "det", frame_id),
            "face": build_frame_key(job_id, "face", frame_id),
        }

    base_path = f"/static/{job_id}"
    return {
        "original": f"{base_path}/original/frame_{frame_id}.jpg",
        "segmentation": f"{base_path}/seg/frame_{frame_id}.jpg",
        "detection": f"{base_path}/det/frame_{frame_id}.jpg",
        "face": f"{base_path}/face/frame_{frame_id}.jpg",
    }


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
    local_path = _build_local_visualization_path(local_dir, job_id, "seg", frame_id)
    _persist_visualization(plot_img, local_path, media_store, job_id, "seg", frame_id)

    # Extract structured data
    if result.masks is None or result.boxes is None:
        return []

    items: list[dict] = []
    names = result.names or {}
    for object_id, (mask_xy, cls_id) in enumerate(
        zip(result.masks.xy, result.boxes.cls.cpu().numpy()),
        start=1,
    ):
        class_name = names.get(int(cls_id), str(int(cls_id)))
        polygon = [[_to_int_coords(x), _to_int_coords(y)] for x, y in mask_xy]
        items.append(
            {
                "object_id": object_id,
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
    local_path = _build_local_visualization_path(local_dir, job_id, "det", frame_id)
    _persist_visualization(plot_img, local_path, media_store, job_id, "det", frame_id)

    # Extract structured data
    if result.boxes is None:
        return []

    items: list[dict] = []
    names = result.names or {}
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()
    for index, box in enumerate(xyxy):
        score = conf[index]
        cls_id = cls_ids[index]
        label = names.get(int(cls_id), str(int(cls_id)))
        items.append(
            {
                "label": label,
                "confidence": float(score),
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
    face_tracker: FaceIdentityTracker | None = None,
) -> list[dict]:
    """Run MTCNN face detection, persist visualization, and return structured data."""
    # Convert BGR (OpenCV) to RGB for MTCNN
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_array)

    # Detect faces — returns (boxes, probs) or (None, None)
    boxes, probs = face_detector.detect(pil_img)

    vis_img = image.copy()

    items: list[dict] = []
    face_id = 0
    used_identities: set[int] = set()
    if boxes is not None and probs is not None:
        for index, box in enumerate(boxes):
            prob = float(probs[index])
            if prob < confidence_threshold:
                continue

            x1 = _to_int_coords(box[0])
            y1 = _to_int_coords(box[1])
            x2 = _to_int_coords(box[2])
            y2 = _to_int_coords(box[3])

            # Draw bounding box on visualization
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face_id += 1
            if face_tracker is not None:
                identity_num = face_tracker.assign_identity(
                    (x1, y1, x2, y2),
                    frame_id=frame_id,
                    used_identities=used_identities,
                )
            else:
                identity_num = face_id
            used_identities.add(identity_num)

            items.append(
                {
                    "face_id": face_id,
                    "identity_id": f"face_{identity_num}",
                    "confidence": prob,
                    "coordinates": [x1, y1, x2, y2],
                }
            )

    local_path = _build_local_visualization_path(local_dir, job_id, "face", frame_id)
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
    face_tracker: FaceIdentityTracker | None = None,
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
        image,
        models.face_detector,
        job_id,
        frame_id,
        local_dir,
        media_store,
        face_tracker=face_tracker,
    )

    files = _build_frame_files(job_id, frame_id, media_store)

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
