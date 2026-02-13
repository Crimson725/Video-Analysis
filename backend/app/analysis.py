"""Frame analysis pipeline: segmentation, detection, face recognition."""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image
from pydantic import ValidationError

from app.face_identity import (
    EdgeFaceTorchEmbedder,
    FaceObservation,
    aggregate_scene_identities,
    stitch_video_identities,
)
from app.models import select_torch_device
from app.schemas import FrameResult
from app.storage import (
    FrameKind,
    MediaStoreError,
    build_analysis_key,
    build_frame_key,
)

if TYPE_CHECKING:
    from app.config import Settings
    from app.storage import MediaStore

logger = logging.getLogger(__name__)


@dataclass
class _FaceTrack:
    """Internal state for a tracked anonymous face identity."""

    identity_num: int
    box: tuple[int, int, int, int]
    last_frame_id: int


@dataclass
class _ObjectTrack:
    """Internal state for a tracked object identity."""

    track_num: int
    label: str
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


class ObjectTrackTracker:
    """Assign stable object track IDs across nearby frames by label + IoU."""

    def __init__(self, iou_threshold: float = 0.25, max_frame_gap: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_frame_gap = max_frame_gap
        self._next_track_num = 1
        self._tracks: dict[int, _ObjectTrack] = {}

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
            track_num
            for track_num, track in self._tracks.items()
            if frame_id - track.last_frame_id > self.max_frame_gap
        ]
        for track_num in stale:
            del self._tracks[track_num]

    def assign_track(
        self,
        label: str,
        box: tuple[int, int, int, int],
        frame_id: int,
        used_track_nums: set[int],
    ) -> int:
        """Return a stable track number for the current object box."""
        self._drop_stale_tracks(frame_id)
        best_track_num: int | None = None
        best_iou = 0.0

        for track_num, track in self._tracks.items():
            if track_num in used_track_nums:
                continue
            if track.label != label:
                continue
            iou = self._intersection_over_union(box, track.box)
            if iou > best_iou:
                best_iou = iou
                best_track_num = track_num

        if best_track_num is not None and best_iou >= self.iou_threshold:
            matched = self._tracks[best_track_num]
            matched.box = box
            matched.last_frame_id = frame_id
            return best_track_num

        track_num = self._next_track_num
        self._next_track_num += 1
        self._tracks[track_num] = _ObjectTrack(
            track_num=track_num,
            label=label,
            box=box,
            last_frame_id=frame_id,
        )
        return track_num


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


def _to_int_box(box: Any) -> tuple[int, int, int, int]:
    """Normalize bounding-box coordinates for JSON payloads and tracker lookups."""
    return (
        _to_int_coords(box[0]),
        _to_int_coords(box[1]),
        _to_int_coords(box[2]),
        _to_int_coords(box[3]),
    )


def _extract_box_ids(boxes: Any) -> Any:
    """Best-effort extraction of detector-provided track IDs."""
    if not hasattr(boxes, "id") or boxes.id is None:
        return None
    try:
        return boxes.id.cpu().numpy()
    except Exception:
        return None


def _resolve_detection_track_num(
    *,
    box_ids: Any,
    index: int,
    object_tracker: ObjectTrackTracker | None,
    label: str,
    box_tuple: tuple[int, int, int, int],
    frame_id: int,
    used_track_nums: set[int],
    job_id: str,
) -> int:
    """Resolve the stable track number using detector IDs, tracker state, or deterministic fallback."""
    if box_ids is not None and index < len(box_ids):
        raw_track = box_ids[index]
        if raw_track is None or (isinstance(raw_track, float) and np.isnan(raw_track)):
            return index + 1
        return int(raw_track)

    if object_tracker is not None:
        return object_tracker.assign_track(
            label=label,
            box=box_tuple,
            frame_id=frame_id,
            used_track_nums=used_track_nums,
        )

    # Deterministic fallback when tracker outputs are unavailable.
    seed = f"{job_id}:{frame_id}:{label}:{index}".encode("utf-8")
    return int(hashlib.sha1(seed).hexdigest()[:8], 16)


def run_detection(
    image: np.ndarray,
    model: Any,
    job_id: str,
    frame_id: int,
    local_dir: str | None,
    media_store: "MediaStore | None" = None,
    object_tracker: ObjectTrackTracker | None = None,
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
    box_ids = _extract_box_ids(result.boxes)
    used_track_nums: set[int] = set()
    for index, box in enumerate(xyxy):
        score = conf[index]
        cls_id = cls_ids[index]
        label = names.get(int(cls_id), str(int(cls_id)))
        box_tuple = _to_int_box(box)
        track_num = _resolve_detection_track_num(
            box_ids=box_ids,
            index=index,
            object_tracker=object_tracker,
            label=label,
            box_tuple=box_tuple,
            frame_id=frame_id,
            used_track_nums=used_track_nums,
            job_id=job_id,
        )

        used_track_nums.add(track_num)
        items.append(
            {
                "track_id": f"{label}_{track_num}",
                "label": label,
                "confidence": float(score),
                "box": list(box_tuple),
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
    for artifact_kind in ("json",):
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
    """Validate payload and persist JSON analysis artifact."""
    try:
        validated = FrameResult.model_validate(frame_payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"Frame payload contract validation failed for frame {frame_id}"
        ) from exc

    json_payload = validated.model_dump_json(by_alias=True).encode("utf-8")
    try:
        media_store.upload_analysis_artifact(job_id, "json", frame_id, json_payload)
    except (MediaStoreError, RuntimeError) as exc:
        _cleanup_analysis_artifacts(media_store, job_id, frame_id)
        raise RuntimeError(
            f"Failed to persist analysis artifacts for frame {frame_id}"
        ) from exc


def _extract_model_provenance(component: str, model: Any, threshold: float | None = None) -> dict[str, Any]:
    """Build a compact model provenance entry for frame metadata."""
    model_id = (
        getattr(model, "model_name", None)
        or getattr(model, "name", None)
        or getattr(model, "__class__", type(model)).__name__
    )
    model_version = (
        getattr(model, "model_version", None)
        or getattr(model, "version", None)
        or getattr(model, "ckpt_path", None)
        or "unknown"
    )
    return {
        "component": str(component),
        "model_id": str(model_id),
        "model_version": str(model_version),
        "threshold": threshold,
    }


def _invoke_optional_enricher(enricher: Any, image: np.ndarray, frame_id: int) -> Any:
    """Invoke optional enrichment callables with flexible signatures."""
    try:
        return enricher(image=image, frame_id=frame_id)
    except TypeError:
        try:
            return enricher(image, frame_id)
        except TypeError:
            return enricher(image)


def _normalize_ocr_blocks(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    blocks: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox") or item.get("box") or [0, 0, 0, 0]
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        blocks.append(
            {
                "text": str(item.get("text", "")).strip(),
                "confidence": float(item.get("confidence", 0.0)),
                "bbox": [_to_int_coords(float(coord)) for coord in bbox],
            }
        )
    return blocks


def _normalize_actions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    actions: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        actions.append(
            {
                "label": str(item.get("label", "unknown")),
                "confidence": float(item.get("confidence", 0.0)),
            }
        )
    return actions


def _normalize_poses(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    poses: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        keypoints_raw = item.get("keypoints", [])
        keypoints: list[dict[str, Any]] = []
        if isinstance(keypoints_raw, list):
            for point in keypoints_raw:
                if not isinstance(point, dict):
                    continue
                keypoints.append(
                    {
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                        "confidence": float(point.get("confidence", 0.0)),
                    }
                )
        poses.append(
            {
                "track_id": str(item.get("track_id", "")) or "unknown_track",
                "confidence": float(item.get("confidence", 0.0)),
                "keypoints": keypoints,
            }
        )
    return poses


def _normalize_camera_motion(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "label": str(raw.get("label", "static")),
        "confidence": float(raw.get("confidence", 0.0)),
    }


def _normalize_quality(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "blur_score": (
            float(raw.get("blur_score"))
            if raw.get("blur_score") is not None
            else None
        ),
        "is_blurry": (
            bool(raw.get("is_blurry")) if raw.get("is_blurry") is not None else None
        ),
        "is_occluded": (
            bool(raw.get("is_occluded")) if raw.get("is_occluded") is not None else None
        ),
    }


def _collect_enrichment_payload(models: Any, image: np.ndarray, frame_id: int) -> dict[str, Any]:
    """Run optional enrichment hooks and normalize outputs."""
    ocr_raw = []
    action_raw = []
    pose_raw = []
    camera_motion_raw = None
    quality_raw = None

    if hasattr(models, "ocr_enricher") and callable(models.ocr_enricher):
        ocr_raw = _invoke_optional_enricher(models.ocr_enricher, image, frame_id)
    if hasattr(models, "action_enricher") and callable(models.action_enricher):
        action_raw = _invoke_optional_enricher(models.action_enricher, image, frame_id)
    if hasattr(models, "pose_enricher") and callable(models.pose_enricher):
        pose_raw = _invoke_optional_enricher(models.pose_enricher, image, frame_id)
    if hasattr(models, "camera_motion_enricher") and callable(models.camera_motion_enricher):
        camera_motion_raw = _invoke_optional_enricher(
            models.camera_motion_enricher, image, frame_id
        )
    if hasattr(models, "quality_enricher") and callable(models.quality_enricher):
        quality_raw = _invoke_optional_enricher(models.quality_enricher, image, frame_id)

    return {
        "ocr_blocks": _normalize_ocr_blocks(ocr_raw),
        "actions": _normalize_actions(action_raw),
        "poses": _normalize_poses(pose_raw),
        "camera_motion": _normalize_camera_motion(camera_motion_raw),
        "quality": _normalize_quality(quality_raw),
    }


def _build_evidence_anchors(
    *,
    frame_id: int,
    timestamp: str,
    analysis_artifact_key: str,
    det_items: list[dict[str, Any]],
    face_items: list[dict[str, Any]],
    enrichment: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build frame-level evidence anchors used by corpus contracts."""
    anchors: list[dict[str, Any]] = []
    for item in det_items:
        anchors.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "artifact_key": analysis_artifact_key,
                "bbox": item.get("box"),
                "text_span": item.get("label"),
            }
        )
    for item in face_items:
        anchors.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "artifact_key": analysis_artifact_key,
                "bbox": item.get("coordinates"),
                "text_span": item.get("identity_id"),
            }
        )
    for block in enrichment.get("ocr_blocks", []):
        anchors.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "artifact_key": analysis_artifact_key,
                "bbox": block.get("bbox"),
                "text_span": block.get("text"),
            }
        )
    return anchors


def _build_frame_metadata(
    *,
    job_id: str,
    frame_id: int,
    timestamp: str,
    source_artifact_key: str,
    models: Any,
    evidence_anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build frame metadata contract for corpus grounding."""
    model_entries = [
        _extract_model_provenance("detector", getattr(models, "detector", None), threshold=0.0),
        _extract_model_provenance("segmenter", getattr(models, "segmenter", None), threshold=0.0),
        _extract_model_provenance("face_detector", getattr(models, "face_detector", None), threshold=0.9),
    ]
    for component in (
        "ocr_enricher",
        "action_enricher",
        "pose_enricher",
        "camera_motion_enricher",
        "quality_enricher",
    ):
        model = getattr(models, component, None)
        if model is not None:
            model_entries.append(_extract_model_provenance(component, model, threshold=None))

    return {
        "provenance": {
            "job_id": job_id,
            "scene_id": None,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "source_artifact_key": source_artifact_key,
        },
        "model_provenance": model_entries,
        "evidence_anchors": evidence_anchors,
    }


def analyze_frame(
    frame_data: dict,
    models: Any,
    job_id: str,
    local_dir: str | None,
    media_store: "MediaStore | None" = None,
    face_tracker: FaceIdentityTracker | None = None,
    object_tracker: ObjectTrackTracker | None = None,
) -> dict:
    """Run all three analysis tasks on a frame and return FrameResult-compatible dict."""
    image = frame_data["image"]
    frame_id = frame_data["frame_id"]
    timestamp = frame_data["timestamp"]

    seg_items = run_segmentation(
        image, models.segmenter, job_id, frame_id, local_dir, media_store
    )
    det_items = run_detection(
        image,
        models.detector,
        job_id,
        frame_id,
        local_dir,
        media_store,
        object_tracker=object_tracker,
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
    enrichment = _collect_enrichment_payload(models, image, frame_id)

    files = _build_frame_files(job_id, frame_id, media_store)
    analysis_key = build_analysis_key(job_id, "json", frame_id)
    evidence_anchors = _build_evidence_anchors(
        frame_id=frame_id,
        timestamp=timestamp,
        analysis_artifact_key=analysis_key,
        det_items=det_items,
        face_items=face_items,
        enrichment=enrichment,
    )
    metadata = _build_frame_metadata(
        job_id=job_id,
        frame_id=frame_id,
        timestamp=timestamp,
        source_artifact_key=files["original"],
        models=models,
        evidence_anchors=evidence_anchors,
    )

    frame_payload: dict[str, Any] = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "files": files,
        "analysis": {
            "semantic_segmentation": seg_items,
            "object_detection": det_items,
            "face_recognition": face_items,
            "enrichment": enrichment,
        },
        "analysis_artifacts": {
            "json": analysis_key,
        },
        "metadata": metadata,
    }

    if media_store is not None:
        _persist_analysis_artifacts(media_store, frame_payload, job_id, frame_id)

    return frame_payload


def _extract_face_observations_from_keyframes(
    *,
    keyframes: list[dict[str, Any]],
    frame_results: list[dict[str, Any]],
    embedder: EdgeFaceTorchEmbedder,
) -> list[FaceObservation]:
    """Convert analyzed keyframe face results into embedding observations."""
    keyframe_by_id = {int(frame.get("frame_id", -1)): frame for frame in keyframes}
    observations: list[FaceObservation] = []
    for frame in frame_results:
        frame_id = int(frame.get("frame_id", -1))
        keyframe = keyframe_by_id.get(frame_id)
        if keyframe is None:
            continue
        scene_id = int(keyframe.get("scene_id", frame_id))
        image = keyframe.get("image")
        if not isinstance(image, np.ndarray):
            continue
        faces = frame.get("analysis", {}).get("face_recognition", [])
        if not isinstance(faces, list):
            continue
        for face in faces:
            if not isinstance(face, dict):
                continue
            coords = face.get("coordinates")
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            face_id = int(face.get("face_id", 0))
            embedding = embedder.embed(image, [int(value) for value in coords])
            observations.append(
                FaceObservation(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    timestamp=str(frame.get("timestamp", "")),
                    face_id=face_id,
                    coordinates=[int(value) for value in coords],
                    confidence=float(face.get("confidence", 0.0)),
                    embedding=embedding,
                    source="keyframe",
                )
            )
    return observations


def _extract_face_observations_from_tracking_frames(
    *,
    tracking_frames: list[dict[str, Any]],
    models: Any,
    job_id: str,
    embedder: EdgeFaceTorchEmbedder,
) -> list[FaceObservation]:
    """Detect faces for sampled tracking frames and convert into observations."""
    observations: list[FaceObservation] = []
    for frame in tracking_frames:
        image = frame.get("image")
        if not isinstance(image, np.ndarray):
            continue
        frame_id = int(frame.get("frame_id", -1))
        scene_id = int(frame.get("scene_id", 0))
        face_items = run_face_recognition(
            image=image,
            face_detector=models.face_detector,
            job_id=job_id,
            frame_id=frame_id,
            local_dir=None,
            media_store=None,
            face_tracker=None,
        )
        for item in face_items:
            coords = item.get("coordinates")
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            embedding = embedder.embed(image, [int(value) for value in coords])
            observations.append(
                FaceObservation(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    timestamp=str(frame.get("timestamp", "")),
                    face_id=int(item.get("face_id", 0)),
                    coordinates=[int(value) for value in coords],
                    confidence=float(item.get("confidence", 0.0)),
                    embedding=embedding,
                    source="tracking",
                )
            )
    return observations


def _apply_identity_metadata_to_keyframes(
    *,
    keyframes: list[dict[str, Any]],
    frame_results: list[dict[str, Any]],
    assignments: dict[tuple[int, int, int], dict[str, Any]],
    scene_to_video: dict[str, dict[str, Any]],
    model_id: str,
) -> None:
    """Attach identity metadata to keyframe face outputs in-place."""
    scene_by_frame_id = {int(frame.get("frame_id", -1)): int(frame.get("scene_id", -1)) for frame in keyframes}
    for frame in frame_results:
        frame_id = int(frame.get("frame_id", -1))
        scene_id = scene_by_frame_id.get(frame_id, frame_id)
        faces = frame.get("analysis", {}).get("face_recognition", [])
        if not isinstance(faces, list):
            continue
        for face in faces:
            if not isinstance(face, dict):
                continue
            face_id = int(face.get("face_id", 0))
            assignment = assignments.get((scene_id, frame_id, face_id))
            if assignment is None:
                face.setdefault("embedding_model_id", model_id)
                continue
            scene_person_id = str(assignment["scene_person_id"])
            video_assignment = scene_to_video.get(scene_person_id, {})
            video_person_id = str(video_assignment.get("video_person_id", "")) or None
            scene_confidence = float(assignment.get("match_confidence", 0.0))
            video_confidence = float(video_assignment.get("confidence", 0.0))
            is_ambiguous = bool(assignment.get("is_identity_ambiguous", False)) or bool(
                video_assignment.get("is_ambiguous", False)
            )
            face["scene_person_id"] = scene_person_id
            face["video_person_id"] = video_person_id
            face["match_confidence"] = max(scene_confidence, video_confidence)
            face["is_identity_ambiguous"] = is_ambiguous
            face["embedding_model_id"] = model_id
            face["identity_id"] = video_person_id or scene_person_id or str(face.get("identity_id", ""))


def run_face_identity_pipeline(
    *,
    keyframes: list[dict[str, Any]],
    frame_results: list[dict[str, Any]],
    tracking_frames: list[dict[str, Any]],
    models: Any,
    settings: "Settings",
    job_id: str,
) -> dict[str, Any]:
    """Run scene-local and video-global face identity aggregation."""
    device = select_torch_device(settings.face_identity_backend)
    embedder = EdgeFaceTorchEmbedder(
        device=device,
        model_id=settings.face_identity_model_id,
        embedding_dimension=settings.face_identity_embedding_dimension,
        weights_path=settings.face_identity_weights_path,
    )

    observations = _extract_face_observations_from_keyframes(
        keyframes=keyframes,
        frame_results=frame_results,
        embedder=embedder,
    )
    observations.extend(
        _extract_face_observations_from_tracking_frames(
            tracking_frames=tracking_frames,
            models=models,
            job_id=job_id,
            embedder=embedder,
        )
    )

    assignments, clusters_by_scene = aggregate_scene_identities(
        observations,
        similarity_threshold=settings.face_identity_scene_similarity_threshold,
        ambiguity_margin=settings.face_identity_ambiguity_margin,
    )
    scene_to_video, video_summary = stitch_video_identities(
        clusters_by_scene,
        similarity_threshold=settings.face_identity_video_similarity_threshold,
        ambiguity_margin=settings.face_identity_ambiguity_margin,
    )

    _apply_identity_metadata_to_keyframes(
        keyframes=keyframes,
        frame_results=frame_results,
        assignments=assignments,
        scene_to_video=scene_to_video,
        model_id=settings.face_identity_model_id,
    )

    scene_summary: list[dict[str, Any]] = []
    for scene_id in sorted(clusters_by_scene):
        for cluster in sorted(clusters_by_scene[scene_id], key=lambda item: item.scene_person_id):
            video_assignment = scene_to_video.get(cluster.scene_person_id, {})
            scene_summary.append(
                {
                    "scene_id": scene_id,
                    "scene_person_id": cluster.scene_person_id,
                    "video_person_id": video_assignment.get("video_person_id"),
                    "confidence": float(video_assignment.get("confidence", 0.0)),
                    "is_ambiguous": bool(video_assignment.get("is_ambiguous", False)),
                    "observation_count": int(cluster.count),
                }
            )

    return {
        "enabled": True,
        "model_id": settings.face_identity_model_id,
        "backend": device.type,
        "scene_identities": scene_summary,
        "video_identities": video_summary,
    }
