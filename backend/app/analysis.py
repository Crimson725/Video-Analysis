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
from ultralytics.utils.plotting import colors as ultralytics_colors

from app.face_identity import (
    ArcFaceRuntimeEmbedder,
    FaceObservation,
    aggregate_scene_identities,
    stitch_video_identities,
)
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


def _rgb_triplet_from_index(index: int) -> list[int]:
    """Resolve deterministic RGB triplet from Ultralytics palette index."""
    color = ultralytics_colors(index, bgr=False)
    return [int(color[0]), int(color[1]), int(color[2])]


def _to_bgr_tuple(color_rgb: list[int]) -> tuple[int, int, int]:
    """Convert RGB triplet metadata to OpenCV-compatible BGR tuple."""
    return (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))


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
            raise RuntimeError(
                f"Failed to encode {frame_kind} frame {frame_id} as JPEG"
            )
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
        cls_index = int(cls_id)
        class_name = names.get(cls_index, str(cls_index))
        polygon = [[_to_int_coords(x), _to_int_coords(y)] for x, y in mask_xy]
        color_rgb = _rgb_triplet_from_index(cls_index)
        items.append(
            {
                "object_id": object_id,
                "class": class_name,
                "mask_polygon": polygon,
                "palette_rgb": list(color_rgb),
                "bbox_rgb": list(color_rgb),
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
        cls_index = int(cls_id)
        label = names.get(cls_index, str(cls_index))
        box_tuple = _to_int_box(box)
        color_rgb = _rgb_triplet_from_index(cls_index)
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
                "palette_rgb": list(color_rgb),
                "bbox_rgb": list(color_rgb),
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
            color_rgb = _rgb_triplet_from_index(identity_num)

            # OpenCV drawing APIs expect BGR channel order.
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), _to_bgr_tuple(color_rgb), 2)
            items.append(
                {
                    "face_id": face_id,
                    "identity_id": f"face_{identity_num}",
                    "confidence": prob,
                    "coordinates": [x1, y1, x2, y2],
                    "palette_rgb": list(color_rgb),
                    "bbox_rgb": list(color_rgb),
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


def _extract_model_provenance(
    component: str, model: Any, threshold: float | None = None
) -> dict[str, Any]:
    """Build a compact model provenance entry for frame metadata."""
    if hasattr(model, "runtime_metadata") and callable(model.runtime_metadata):
        metadata = model.runtime_metadata()
        model_id = (
            metadata.get("model_name")
            or metadata.get("model_id")
            or getattr(model, "__class__", type(model)).__name__
        )
        provider_path = metadata.get("provider_path")
        model_version = metadata.get("backend") or "unknown"
        payload = {
            "component": str(component),
            "model_id": str(model_id),
            "model_version": str(model_version),
            "threshold": threshold,
        }
        if isinstance(provider_path, list):
            payload["provider_path"] = list(provider_path)
        return payload

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
            float(raw.get("blur_score")) if raw.get("blur_score") is not None else None
        ),
        "is_blurry": (
            bool(raw.get("is_blurry")) if raw.get("is_blurry") is not None else None
        ),
        "is_occluded": (
            bool(raw.get("is_occluded")) if raw.get("is_occluded") is not None else None
        ),
    }


def _collect_enrichment_payload(
    models: Any, image: np.ndarray, frame_id: int
) -> dict[str, Any]:
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
    if hasattr(models, "camera_motion_enricher") and callable(
        models.camera_motion_enricher
    ):
        camera_motion_raw = _invoke_optional_enricher(
            models.camera_motion_enricher, image, frame_id
        )
    if hasattr(models, "quality_enricher") and callable(models.quality_enricher):
        quality_raw = _invoke_optional_enricher(
            models.quality_enricher, image, frame_id
        )

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
    raw_frame_index: int | None,
    source_artifact_key: str,
    models: Any,
    evidence_anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build frame metadata contract for corpus grounding."""
    model_entries = [
        _extract_model_provenance(
            "detector", getattr(models, "detector", None), threshold=0.0
        ),
        _extract_model_provenance(
            "segmenter", getattr(models, "segmenter", None), threshold=0.0
        ),
        _extract_model_provenance(
            "face_detector", getattr(models, "face_detector", None), threshold=0.9
        ),
        _extract_model_provenance(
            "face_embedder", getattr(models, "face_embedder", None), threshold=None
        ),
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
            model_entries.append(
                _extract_model_provenance(component, model, threshold=None)
            )

    return {
        "provenance": {
            "job_id": job_id,
            "scene_id": None,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "raw_frame_index": raw_frame_index,
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
    raw_frame_index = frame_data.get("raw_frame_index")
    if not isinstance(raw_frame_index, int) or isinstance(raw_frame_index, bool):
        raw_frame_index = None

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
        raw_frame_index=raw_frame_index,
        source_artifact_key=files["original"],
        models=models,
        evidence_anchors=evidence_anchors,
    )

    frame_payload: dict[str, Any] = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "raw_frame_index": raw_frame_index,
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
    embedder: Any,
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
    embedder: Any,
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
    scene_by_frame_id = {
        int(frame.get("frame_id", -1)): int(frame.get("scene_id", -1))
        for frame in keyframes
    }
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
            face["identity_id"] = (
                video_person_id or scene_person_id or str(face.get("identity_id", ""))
            )


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
    embedder = getattr(models, "face_embedder", None)
    if embedder is None or not hasattr(embedder, "embed"):
        embedder = ArcFaceRuntimeEmbedder(
            model_name=settings.face_identity_arcface_model_name,
            provider_order=settings.face_identity_arcface_provider_order,
            fallback_behavior=settings.face_identity_arcface_fallback_behavior,
            embedding_dimension=settings.face_identity_embedding_dimension,
        )

    runtime_metadata: dict[str, Any] = {}
    if hasattr(embedder, "runtime_metadata") and callable(embedder.runtime_metadata):
        metadata_raw = embedder.runtime_metadata()
        if isinstance(metadata_raw, dict):
            runtime_metadata = dict(metadata_raw)
    provider_path = runtime_metadata.get("provider_path", [])
    logger.info(
        "Face identity pipeline starting model=%s backend=%s provider_path=%s",
        settings.face_identity_arcface_model_name,
        runtime_metadata.get("backend", "arcface"),
        provider_path,
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
        model_id=settings.face_identity_arcface_model_name,
    )

    scene_summary: list[dict[str, Any]] = []
    for scene_id in sorted(clusters_by_scene):
        for cluster in sorted(
            clusters_by_scene[scene_id], key=lambda item: item.scene_person_id
        ):
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
        "model_id": settings.face_identity_arcface_model_name,
        "backend": runtime_metadata.get("backend", "arcface"),
        "provider_path": provider_path if isinstance(provider_path, list) else [],
        "active_provider": runtime_metadata.get("active_provider"),
        "scene_identities": scene_summary,
        "video_identities": video_summary,
    }


def _timestamp_seconds(timestamp: Any) -> float:
    """Best-effort conversion for HH:MM:SS.mmm timestamp strings."""
    text = str(timestamp or "").strip()
    parts = text.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return 0.0
    return max(0.0, hours * 3600 + minutes * 60 + seconds)


def _coerce_int_box(value: Any) -> list[int] | None:
    """Normalize a box payload to [x1, y1, x2, y2] ints."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _box_intersection_area(box_a: list[int], box_b: list[int]) -> int:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_iou(box_a: list[int], box_b: list[int]) -> float:
    inter = _box_intersection_area(box_a, box_b)
    if inter <= 0:
        return 0.0
    union = _box_area(box_a) + _box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _box_center_inside(inner: list[int], outer: list[int]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _face_person_affinity(face_box: list[int], person_box: list[int]) -> float:
    """Score face↔person compatibility with precision-biased geometry checks."""
    face_area = _box_area(face_box)
    if face_area <= 0:
        return 0.0
    overlap_ratio = _box_intersection_area(face_box, person_box) / face_area
    center_inside = 1.0 if _box_center_inside(face_box, person_box) else 0.0
    iou = _box_iou(face_box, person_box)
    return 0.55 * overlap_ratio + 0.35 * center_inside + 0.10 * iou


def _resolve_face_identity(
    face: dict[str, Any],
) -> tuple[str | None, str | None, float]:
    """Resolve identity tuple ordered by strongest available continuity signal."""
    video_person_id = str(face.get("video_person_id", "")).strip()
    scene_person_id = str(face.get("scene_person_id", "")).strip()
    identity_id = str(face.get("identity_id", "")).strip()
    confidence = float(face.get("match_confidence") or face.get("confidence") or 0.0)
    if video_person_id:
        return video_person_id, "video_person_id", confidence
    if scene_person_id:
        return scene_person_id, "scene_person_id", confidence
    if identity_id:
        return identity_id, "identity_id", confidence
    return None, None, 0.0


def _deterministic_person_track_id(job_id: str, key: str) -> str:
    seed = f"{job_id}:{key}".encode("utf-8")
    return f"person_track_{hashlib.sha1(seed).hexdigest()[:16]}"


def _deterministic_object_track_id(job_id: str, key: str) -> str:
    seed = f"{job_id}:{key}".encode("utf-8")
    return f"object_track_{hashlib.sha1(seed).hexdigest()[:16]}"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _object_track_continuity_score(
    *,
    group: dict[str, Any],
    slot: dict[str, Any],
) -> float:
    group_first_frame = int(group.get("first_frame_id") or 0)
    group_last_frame = int(group.get("last_frame_id") or 0)
    slot_first_frame = int(slot.get("first_frame_id") or 0)
    slot_last_frame = int(slot.get("last_frame_id") or 0)

    group_first_box = _coerce_int_box(group.get("first_box"))
    group_last_box = _coerce_int_box(group.get("last_box"))
    slot_first_box = _coerce_int_box(slot.get("first_box"))
    slot_last_box = _coerce_int_box(slot.get("last_box"))
    if (
        group_first_box is None
        or group_last_box is None
        or slot_first_box is None
        or slot_last_box is None
    ):
        return 0.0

    frame_gap = 0
    iou = 0.0
    if slot_first_frame >= group_last_frame:
        frame_gap = slot_first_frame - group_last_frame
        iou = _box_iou(slot_first_box, group_last_box)
    elif group_first_frame >= slot_last_frame:
        frame_gap = group_first_frame - slot_last_frame
        iou = _box_iou(group_first_box, slot_last_box)
    else:
        frame_gap = 0
        iou = max(
            _box_iou(slot_first_box, group_first_box),
            _box_iou(slot_last_box, group_last_box),
        )

    group_timestamps = group.get("timestamps", [])
    slot_timestamps = slot.get("timestamps", [])
    time_gap = 0.0
    if group_timestamps and slot_timestamps:
        group_first_t = float(group_timestamps[0])
        group_last_t = float(group_timestamps[-1])
        slot_first_t = float(slot_timestamps[0])
        slot_last_t = float(slot_timestamps[-1])
        if slot_first_t >= group_last_t:
            time_gap = slot_first_t - group_last_t
        elif group_first_t >= slot_last_t:
            time_gap = group_first_t - slot_last_t
        else:
            time_gap = 0.0

    if frame_gap > 45 and time_gap > 4.5:
        return 0.0

    group_area = _box_area(group_last_box)
    slot_area = _box_area(slot_first_box)
    if group_area <= 0 or slot_area <= 0:
        size_score = 0.0
    else:
        size_score = min(group_area, slot_area) / max(group_area, slot_area)

    temporal_score = max(0.0, 1.0 - min(1.0, time_gap / 4.5))
    return (0.55 * iou) + (0.25 * temporal_score) + (0.20 * size_score)


def run_object_tracking_summary(
    *,
    frame_results: list[dict[str, Any]],
    job_id: str,
    max_evidence_per_track: int = 25,
    merge_threshold: float = 0.42,
    ambiguity_margin: float = 0.04,
) -> dict[str, Any]:
    """Aggregate per-frame detections into deterministic canonical object tracks."""
    evidence_limit = max(1, int(max_evidence_per_track))
    ordered_frames = sorted(
        frame_results, key=lambda item: int(item.get("frame_id", 0))
    )
    track_slots: dict[str, dict[str, Any]] = {}

    for frame in ordered_frames:
        frame_id = int(frame.get("frame_id", 0))
        timestamp = str(frame.get("timestamp", ""))
        timestamp_sec = _timestamp_seconds(timestamp)
        analysis = frame.get("analysis", {})
        detections_raw = analysis.get("object_detection", [])
        if not isinstance(detections_raw, list):
            detections_raw = []

        for index, detection in enumerate(detections_raw):
            if not isinstance(detection, dict):
                continue
            label = str(detection.get("label", "")).strip() or "unknown"
            label_key = label.lower()
            track_id = str(detection.get("track_id", "")).strip()
            if not track_id:
                track_id = f"{label_key}_{frame_id}_{index + 1}"
                detection["track_id"] = track_id

            slot_key = f"{label_key}::{track_id}"
            slot = track_slots.setdefault(
                slot_key,
                {
                    "label": label,
                    "source_track_id": track_id,
                    "frame_ids": [],
                    "timestamps": [],
                    "confidences": [],
                    "evidence": [],
                    "detections": [],
                },
            )
            slot["frame_ids"].append(frame_id)
            slot["timestamps"].append(timestamp_sec)
            slot["confidences"].append(
                _coerce_float(detection.get("confidence"), default=0.0)
            )
            slot["detections"].append(detection)
            slot["evidence"].append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "detection_index": index,
                    "label": label,
                    "track_id": track_id,
                    "confidence": _coerce_float(
                        detection.get("confidence"), default=0.0
                    ),
                    "box": _coerce_int_box(detection.get("box")),
                }
            )

    slot_records: list[dict[str, Any]] = []
    for slot_key in sorted(track_slots):
        slot = track_slots[slot_key]
        evidence = sorted(
            slot["evidence"],
            key=lambda item: (
                int(item.get("frame_id", 0)),
                int(item.get("detection_index", 0)),
                str(item.get("track_id", "")),
            ),
        )
        frame_ids = sorted({int(value) for value in slot["frame_ids"]})
        timestamps = sorted(float(value) for value in slot["timestamps"])
        first_box = next(
            (item.get("box") for item in evidence if _coerce_int_box(item.get("box"))),
            None,
        )
        last_box = next(
            (
                item.get("box")
                for item in reversed(evidence)
                if _coerce_int_box(item.get("box"))
            ),
            None,
        )
        slot_records.append(
            {
                "slot_key": slot_key,
                "label": str(slot["label"]),
                "label_key": str(slot["label"]).lower(),
                "source_track_id": str(slot["source_track_id"]),
                "frame_ids": frame_ids,
                "timestamps": timestamps,
                "confidences": [float(value) for value in slot["confidences"]],
                "evidence": evidence,
                "detections": list(slot["detections"]),
                "first_frame_id": frame_ids[0] if frame_ids else None,
                "last_frame_id": frame_ids[-1] if frame_ids else None,
                "first_box": first_box,
                "last_box": last_box,
            }
        )

    canonical_groups: list[dict[str, Any]] = []
    resolved_threshold = max(0.0, min(1.0, float(merge_threshold)))
    resolved_ambiguity = max(0.0, float(ambiguity_margin))
    for slot in sorted(
        slot_records,
        key=lambda item: (
            int(item["first_frame_id"] or 0),
            str(item["label_key"]),
            str(item["source_track_id"]),
        ),
    ):
        candidates: list[tuple[float, int]] = []
        for group_index, group in enumerate(canonical_groups):
            if str(group["label_key"]) != str(slot["label_key"]):
                continue
            score = _object_track_continuity_score(group=group, slot=slot)
            if score <= 0.0:
                continue
            candidates.append((score, group_index))
        candidates.sort(key=lambda item: (-float(item[0]), int(item[1])))

        selected_group_index: int | None = None
        selected_score = 0.0
        is_slot_ambiguous = False
        if candidates:
            selected_score, selected_group_index = candidates[0]
            second_score = candidates[1][0] if len(candidates) > 1 else 0.0
            if selected_score < resolved_threshold:
                selected_group_index = None
            elif (
                len(candidates) > 1
                and (selected_score - second_score) <= resolved_ambiguity
            ):
                selected_group_index = None
                is_slot_ambiguous = True

        if selected_group_index is None:
            canonical_groups.append(
                {
                    "label": slot["label"],
                    "label_key": slot["label_key"],
                    "source_track_ids": [slot["source_track_id"]],
                    "frame_ids": list(slot["frame_ids"]),
                    "timestamps": list(slot["timestamps"]),
                    "confidences": list(slot["confidences"]),
                    "evidence": list(slot["evidence"]),
                    "detections": list(slot["detections"]),
                    "first_frame_id": slot["first_frame_id"],
                    "last_frame_id": slot["last_frame_id"],
                    "first_box": slot["first_box"],
                    "last_box": slot["last_box"],
                    "merge_confidence_values": [1.0],
                    "is_identity_ambiguous": bool(is_slot_ambiguous),
                }
            )
            continue

        group = canonical_groups[selected_group_index]
        group["source_track_ids"].append(slot["source_track_id"])
        group["frame_ids"].extend(slot["frame_ids"])
        group["timestamps"].extend(slot["timestamps"])
        group["confidences"].extend(slot["confidences"])
        group["evidence"].extend(slot["evidence"])
        group["detections"].extend(slot["detections"])
        if (
            group["first_frame_id"] is None
            or (
                slot["first_frame_id"] is not None
                and int(slot["first_frame_id"]) < int(group["first_frame_id"])
            )
        ):
            group["first_frame_id"] = slot["first_frame_id"]
            group["first_box"] = slot["first_box"]
        if (
            group["last_frame_id"] is None
            or (
                slot["last_frame_id"] is not None
                and int(slot["last_frame_id"]) > int(group["last_frame_id"])
            )
        ):
            group["last_frame_id"] = slot["last_frame_id"]
            group["last_box"] = slot["last_box"]
        group["merge_confidence_values"].append(round(float(selected_score), 4))

    tracks: list[dict[str, Any]] = []
    for group in sorted(
        canonical_groups,
        key=lambda item: (
            int(item.get("first_frame_id") or 0),
            str(item.get("label_key")),
            sorted(set(str(value) for value in item.get("source_track_ids", []))),
        ),
    ):
        source_track_ids = sorted(
            set(str(track_id) for track_id in group.get("source_track_ids", []))
        )
        confidences = [float(value) for value in group.get("confidences", [])]
        frame_ids = sorted(set(int(value) for value in group.get("frame_ids", [])))
        timestamps = sorted(float(value) for value in group.get("timestamps", []))
        stable_key = f"{group['label_key']}::{','.join(source_track_ids)}"
        object_track_id = _deterministic_object_track_id(job_id, stable_key)
        merge_confidences = [
            float(value) for value in group.get("merge_confidence_values", []) if value
        ]
        identity_confidence: float | None = None
        if merge_confidences:
            identity_confidence = round(
                sum(merge_confidences) / len(merge_confidences),
                4,
            )

        evidence = sorted(
            group.get("evidence", []),
            key=lambda item: (
                int(item.get("frame_id", 0)),
                int(item.get("detection_index", 0)),
                str(item.get("track_id", "")),
            ),
        )[:evidence_limit]

        for detection in group.get("detections", []):
            detection["object_track_id"] = object_track_id
            detection["object_identity_confidence"] = identity_confidence
            detection["object_identity_is_ambiguous"] = bool(
                group.get("is_identity_ambiguous", False)
            )

        sample_count = len(confidences)
        mean_conf = (sum(confidences) / sample_count) if sample_count else 0.0
        max_conf = max(confidences) if sample_count else 0.0
        min_conf = min(confidences) if sample_count else 0.0
        source_track_id = source_track_ids[0] if source_track_ids else ""

        tracks.append(
            {
                "object_track_id": object_track_id,
                "label": str(group.get("label", "unknown")),
                "source_track_id": source_track_id,
                "source_track_ids": source_track_ids,
                "identity_confidence": identity_confidence,
                "is_identity_ambiguous": bool(
                    group.get("is_identity_ambiguous", False)
                ),
                "confidence": {
                    "mean": round(mean_conf, 4),
                    "max": round(max_conf, 4),
                    "min": round(min_conf, 4),
                    "samples": sample_count,
                },
                "frame_span": {
                    "first_frame_id": frame_ids[0] if frame_ids else None,
                    "last_frame_id": frame_ids[-1] if frame_ids else None,
                    "observation_count": len(frame_ids),
                },
                "temporal_span": {
                    "first_seen": timestamps[0] if timestamps else 0.0,
                    "last_seen": timestamps[-1] if timestamps else 0.0,
                    "duration_sec": (timestamps[-1] - timestamps[0])
                    if len(timestamps) >= 2
                    else 0.0,
                },
                "evidence": evidence,
            }
        )

    tracks.sort(
        key=lambda item: (
            int(item["frame_span"]["first_frame_id"] or 0),
            str(item["label"]).lower(),
            str(item["source_track_id"]),
        )
    )
    return {
        "enabled": True,
        "method": "object_tracking_v1",
        "tracks": tracks,
    }


def _new_person_track_slot(track_id: str) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "frame_ids": [],
        "timestamps": [],
        "identity_votes": {},
        "identity_source_votes": {},
        "evidence": [],
        "observations": [],
    }


def _collect_person_detections(
    *,
    detections_raw: Any,
    frame_id: int,
    timestamp_sec: float,
    track_slots: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], list[int], str]]:
    person_detections: list[tuple[dict[str, Any], list[int], str]] = []
    if not isinstance(detections_raw, list):
        return person_detections

    for index, detection in enumerate(detections_raw):
        if not isinstance(detection, dict):
            continue
        label = str(detection.get("label", "")).strip().lower()
        if label != "person":
            continue
        box = _coerce_int_box(detection.get("box"))
        if box is None:
            continue

        track_id = (
            str(detection.get("track_id", "")).strip()
            or f"person_track_{frame_id}_{index + 1}"
        )
        detection["track_id"] = track_id
        person_detections.append((detection, box, track_id))

        slot = track_slots.setdefault(track_id, _new_person_track_slot(track_id))
        slot["frame_ids"].append(frame_id)
        slot["timestamps"].append(timestamp_sec)
        slot["observations"].append(
            {
                "frame_id": frame_id,
                "timestamp_sec": timestamp_sec,
                "box": list(box),
            }
        )

    return person_detections


def _best_person_detection_match(
    face_box: list[int],
    person_detections: list[tuple[dict[str, Any], list[int], str]],
) -> tuple[dict[str, Any], list[int], str, float] | None:
    best_match: tuple[dict[str, Any], list[int], str] | None = None
    best_score = 0.0
    for detection, person_box, track_id in person_detections:
        score = _face_person_affinity(face_box, person_box)
        if score > best_score:
            best_score = score
            best_match = (detection, person_box, track_id)

    if best_match is None:
        return None
    return best_match[0], best_match[1], best_match[2], best_score


def _apply_identity_vote(
    *,
    slot: dict[str, Any],
    detection: dict[str, Any],
    person_box: list[int],
    face: dict[str, Any],
    face_box: list[int],
    frame_id: int,
    timestamp: str,
    identity_id: str,
    source: str,
    identity_confidence: float,
    association_confidence: float,
) -> None:
    vote_slot = slot["identity_votes"].setdefault(
        identity_id,
        {
            "count": 0,
            "confidence_sum": 0.0,
        },
    )
    vote_slot["count"] += 1
    vote_slot["confidence_sum"] += identity_confidence

    source_votes = slot["identity_source_votes"]
    source_votes[source] = int(source_votes.get(source, 0)) + 1
    slot["evidence"].append(
        {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "object_track_id": slot["track_id"],
            "face_id": int(face.get("face_id", 0)),
            "identity_id": identity_id,
            "identity_source": source,
            "identity_confidence": float(identity_confidence),
            "association_confidence": round(association_confidence, 4),
            "face_box": face_box,
            "person_box": person_box,
        }
    )
    detection["person_identity_candidate"] = identity_id


def _resolve_track_identity(
    slot: dict[str, Any],
    *,
    ambiguity_margin: float,
) -> tuple[str | None, str | None, float | None, bool]:
    votes: dict[str, dict[str, Any]] = slot["identity_votes"]
    resolved_identity: str | None = None
    resolved_source: str | None = None
    resolved_confidence: float | None = None
    is_ambiguous = False

    if not votes:
        return resolved_identity, resolved_source, resolved_confidence, is_ambiguous

    ranked = sorted(
        votes.items(),
        key=lambda item: (
            -int(item[1]["count"]),
            -(float(item[1]["confidence_sum"]) / max(1, int(item[1]["count"]))),
            item[0],
        ),
    )
    top_identity, top_stats = ranked[0]
    top_count = int(top_stats["count"])
    top_avg_conf = float(top_stats["confidence_sum"]) / max(1, top_count)
    total_votes = sum(int(v["count"]) for v in votes.values())
    dominance = top_count / max(1, total_votes)

    if len(ranked) > 1:
        second_identity, second_stats = ranked[1]
        second_count = int(second_stats["count"])
        second_avg_conf = float(second_stats["confidence_sum"]) / max(1, second_count)
        is_ambiguous = (
            second_count >= max(1, top_count - 1)
            and (top_avg_conf - second_avg_conf) <= max(0.0, ambiguity_margin)
        )
        if second_identity == top_identity:
            is_ambiguous = False

    if not is_ambiguous and (
        dominance >= 0.6 or top_count >= 2 or top_avg_conf >= 0.85
    ):
        resolved_identity = top_identity
        source_votes = slot["identity_source_votes"]
        resolved_source = sorted(
            source_votes.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[0][0]
        resolved_confidence = round(min(1.0, 0.5 * dominance + 0.5 * top_avg_conf), 4)

    return resolved_identity, resolved_source, resolved_confidence, is_ambiguous


def _normalize_track_slots(
    track_slots: dict[str, dict[str, Any]],
    *,
    ambiguity_margin: float,
) -> list[dict[str, Any]]:
    normalized_tracks: list[dict[str, Any]] = []
    for track_id in sorted(track_slots):
        slot = track_slots[track_id]
        (
            resolved_identity,
            resolved_source,
            resolved_confidence,
            is_ambiguous,
        ) = _resolve_track_identity(slot, ambiguity_margin=ambiguity_margin)
        observations = sorted(
            [
                obs
                for obs in slot.get("observations", [])
                if isinstance(obs, dict)
                and isinstance(obs.get("box"), list)
                and len(obs.get("box")) == 4
            ],
            key=lambda item: (
                int(item.get("frame_id", 0)),
                float(item.get("timestamp_sec", 0.0)),
            ),
        )
        first_box = observations[0]["box"] if observations else None
        last_box = observations[-1]["box"] if observations else None

        normalized_tracks.append(
            {
                "track_id": track_id,
                "frame_ids": sorted({int(frame_id) for frame_id in slot["frame_ids"]}),
                "timestamps": sorted(
                    float(timestamp) for timestamp in slot["timestamps"]
                ),
                "identity_id": resolved_identity,
                "identity_source": resolved_source,
                "identity_confidence": resolved_confidence,
                "is_identity_ambiguous": is_ambiguous,
                "first_box": first_box,
                "last_box": last_box,
                "observations": observations,
                "evidence": sorted(
                    slot["evidence"],
                    key=lambda item: (
                        int(item.get("frame_id", 0)),
                        int(item.get("face_id", 0)),
                        str(item.get("object_track_id", "")),
                    ),
                ),
            }
        )
    return normalized_tracks


def _person_track_continuity_score(
    *,
    source_track: dict[str, Any],
    candidate_track: dict[str, Any],
) -> float:
    source_frame_ids = source_track.get("frame_ids", [])
    candidate_frame_ids = candidate_track.get("frame_ids", [])
    if not source_frame_ids or not candidate_frame_ids:
        return 0.0
    source_first = int(source_frame_ids[0])
    source_last = int(source_frame_ids[-1])
    candidate_first = int(candidate_frame_ids[0])
    candidate_last = int(candidate_frame_ids[-1])
    source_first_box = _coerce_int_box(source_track.get("first_box"))
    source_last_box = _coerce_int_box(source_track.get("last_box"))
    candidate_first_box = _coerce_int_box(candidate_track.get("first_box"))
    candidate_last_box = _coerce_int_box(candidate_track.get("last_box"))
    if (
        source_first_box is None
        or source_last_box is None
        or candidate_first_box is None
        or candidate_last_box is None
    ):
        return 0.0

    frame_gap = 0
    iou = 0.0
    if source_first >= candidate_last:
        frame_gap = source_first - candidate_last
        iou = _box_iou(source_first_box, candidate_last_box)
    elif candidate_first >= source_last:
        frame_gap = candidate_first - source_last
        iou = _box_iou(candidate_first_box, source_last_box)
    else:
        frame_gap = 0
        iou = max(
            _box_iou(source_first_box, candidate_first_box),
            _box_iou(source_last_box, candidate_last_box),
        )

    time_gap = 0.0
    source_timestamps = source_track.get("timestamps", [])
    candidate_timestamps = candidate_track.get("timestamps", [])
    if source_timestamps and candidate_timestamps:
        source_start = float(source_timestamps[0])
        source_end = float(source_timestamps[-1])
        candidate_start = float(candidate_timestamps[0])
        candidate_end = float(candidate_timestamps[-1])
        if source_start >= candidate_end:
            time_gap = source_start - candidate_end
        elif candidate_start >= source_end:
            time_gap = candidate_start - source_end
        else:
            time_gap = 0.0

    if frame_gap > 30 and time_gap > 3.5:
        return 0.0
    temporal_score = max(0.0, 1.0 - min(1.0, time_gap / 3.5))
    return (0.7 * iou) + (0.3 * temporal_score)


def _link_unresolved_tracks_by_continuity(
    normalized_tracks: list[dict[str, Any]],
    *,
    ambiguity_margin: float,
) -> None:
    resolved_tracks = [
        track for track in normalized_tracks if str(track.get("identity_id") or "").strip()
    ]

    for track in sorted(
        normalized_tracks,
        key=lambda item: (
            int(item.get("frame_ids", [0])[0]) if item.get("frame_ids") else 0,
            str(item.get("track_id", "")),
        ),
    ):
        if str(track.get("identity_id") or "").strip():
            continue

        source_frame_ids = track.get("frame_ids", [])
        source_first_frame = (
            int(source_frame_ids[0]) if isinstance(source_frame_ids, list) and source_frame_ids else 0
        )
        source_last_frame = (
            int(source_frame_ids[-1]) if isinstance(source_frame_ids, list) and source_frame_ids else source_first_frame
        )
        preceding_candidates = [
            candidate
            for candidate in resolved_tracks
            if candidate.get("frame_ids")
            and int(candidate["frame_ids"][-1]) <= source_first_frame
        ]
        following_candidates = [
            candidate
            for candidate in resolved_tracks
            if candidate.get("frame_ids")
            and int(candidate["frame_ids"][0]) >= source_last_frame
        ]
        if preceding_candidates:
            candidate_pool = preceding_candidates
        elif following_candidates:
            candidate_pool = following_candidates
        else:
            candidate_pool = resolved_tracks

        candidates: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidate_pool:
            candidate_identity = str(candidate.get("identity_id") or "").strip()
            if not candidate_identity:
                continue
            score = _person_track_continuity_score(
                source_track=track,
                candidate_track=candidate,
            )
            if score <= 0.0:
                continue
            candidates.append((score, candidate))

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: (
                -float(item[0]),
                str(item[1].get("identity_id") or ""),
                str(item[1].get("track_id") or ""),
            )
        )
        best_score, best_candidate = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        if best_score < 0.45:
            continue
        if len(candidates) > 1 and (best_score - second_score) <= ambiguity_margin:
            track["is_identity_ambiguous"] = True
            continue

        inherited_conf = _coerce_float(best_candidate.get("identity_confidence"), 0.0)
        continuity_confidence = round(
            min(1.0, (0.5 * best_score) + (0.5 * inherited_conf)),
            4,
        )
        track["identity_id"] = best_candidate.get("identity_id")
        track["identity_source"] = "track_continuity"
        track["identity_confidence"] = continuity_confidence
        resolved_tracks.append(track)


def _build_fused_groups(
    normalized_tracks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    fused_groups: dict[str, dict[str, Any]] = {}
    for track in normalized_tracks:
        identity_id = track["identity_id"]
        group_key = (
            f"identity::{identity_id}" if identity_id else f"track::{track['track_id']}"
        )
        group = fused_groups.setdefault(
            group_key,
            {
                "identity_id": identity_id,
                "identity_source": track["identity_source"],
                "identity_confidence_values": [],
                "is_identity_ambiguous": False,
                "object_track_ids": [],
                "frame_ids": [],
                "timestamps": [],
                "evidence": [],
            },
        )
        group["object_track_ids"].append(track["track_id"])
        group["frame_ids"].extend(track["frame_ids"])
        group["timestamps"].extend(track["timestamps"])
        group["evidence"].extend(track["evidence"])
        if track["identity_confidence"] is not None:
            group["identity_confidence_values"].append(
                float(track["identity_confidence"])
            )
        group["is_identity_ambiguous"] = bool(group["is_identity_ambiguous"]) or bool(
            track["is_identity_ambiguous"]
        )
    return fused_groups


def _build_fused_tracks(
    *,
    fused_groups: dict[str, dict[str, Any]],
    job_id: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    fused_tracks: list[dict[str, Any]] = []
    track_lookup: dict[str, dict[str, Any]] = {}
    for group_key in sorted(fused_groups):
        group = fused_groups[group_key]
        object_track_ids = sorted(
            set(str(track_id) for track_id in group["object_track_ids"])
        )
        frame_ids = sorted(set(int(frame_id) for frame_id in group["frame_ids"]))
        timestamps = sorted(float(timestamp) for timestamp in group["timestamps"])

        identity_id = group["identity_id"]
        stable_key = (
            f"identity::{identity_id}"
            if identity_id
            else f"tracks::{','.join(object_track_ids)}"
        )
        person_track_id = _deterministic_person_track_id(job_id, stable_key)

        identity_conf_values = group["identity_confidence_values"]
        identity_confidence: float | None = None
        if identity_conf_values:
            identity_confidence = round(
                sum(identity_conf_values) / len(identity_conf_values), 4
            )

        payload = {
            "person_track_id": person_track_id,
            "identity_id": identity_id,
            "identity_source": group["identity_source"],
            "identity_confidence": identity_confidence,
            "is_identity_ambiguous": bool(group["is_identity_ambiguous"]),
            "object_track_ids": object_track_ids,
            "frame_span": {
                "first_frame_id": frame_ids[0] if frame_ids else None,
                "last_frame_id": frame_ids[-1] if frame_ids else None,
                "observation_count": len(frame_ids),
            },
            "temporal_span": {
                "first_seen": timestamps[0] if timestamps else 0.0,
                "last_seen": timestamps[-1] if timestamps else 0.0,
                "duration_sec": (timestamps[-1] - timestamps[0])
                if len(timestamps) >= 2
                else 0.0,
            },
            "evidence": sorted(
                group["evidence"],
                key=lambda item: (
                    int(item.get("frame_id", 0)),
                    int(item.get("face_id", 0)),
                    str(item.get("object_track_id", "")),
                ),
            )[:25],
        }
        fused_tracks.append(payload)
        for object_track_id in object_track_ids:
            track_lookup[object_track_id] = payload

    return fused_tracks, track_lookup


def _annotate_fused_person_tracks(
    *,
    ordered_frames: list[dict[str, Any]],
    track_lookup: dict[str, dict[str, Any]],
) -> None:
    for frame in ordered_frames:
        analysis = frame.get("analysis", {})
        detections_raw = analysis.get("object_detection", [])
        if not isinstance(detections_raw, list):
            continue

        for detection in detections_raw:
            if not isinstance(detection, dict):
                continue
            label = str(detection.get("label", "")).strip().lower()
            if label != "person":
                continue
            track_id = str(detection.get("track_id", "")).strip()
            if not track_id:
                continue

            resolved = track_lookup.get(track_id)
            if resolved is None:
                continue
            detection["person_track_id"] = resolved["person_track_id"]
            detection["person_identity_id"] = resolved["identity_id"]
            detection["person_identity_source"] = resolved["identity_source"]
            detection["person_identity_confidence"] = resolved["identity_confidence"]
            detection["person_identity_is_ambiguous"] = bool(
                resolved.get("is_identity_ambiguous", False)
            )
            detection.pop("person_identity_candidate", None)


def run_person_tracking_fusion(
    *,
    frame_results: list[dict[str, Any]],
    job_id: str,
    ambiguity_margin: float = 0.03,
) -> dict[str, Any]:
    """Fuse person object tracks with face identity evidence into stable video tracks."""
    resolved_ambiguity_margin = max(0.0, float(ambiguity_margin))
    track_slots: dict[str, dict[str, Any]] = {}
    ordered_frames = sorted(
        frame_results, key=lambda item: int(item.get("frame_id", 0))
    )

    for frame in ordered_frames:
        frame_id = int(frame.get("frame_id", 0))
        timestamp = str(frame.get("timestamp", ""))
        timestamp_sec = _timestamp_seconds(timestamp)
        analysis = frame.get("analysis", {})
        person_detections = _collect_person_detections(
            detections_raw=analysis.get("object_detection", []),
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            track_slots=track_slots,
        )
        faces_raw = analysis.get("face_recognition", [])
        if not isinstance(faces_raw, list):
            continue

        for face in faces_raw:
            if not isinstance(face, dict):
                continue
            face_box = _coerce_int_box(face.get("coordinates"))
            if face_box is None:
                continue

            identity_id, source, identity_confidence = _resolve_face_identity(face)
            if identity_id is None or source is None:
                continue

            match = _best_person_detection_match(face_box, person_detections)
            # Precision-first threshold keeps identity binding conservative.
            if match is None or match[3] < 0.45:
                continue

            (
                matched_detection,
                matched_person_box,
                matched_track_id,
                association_confidence,
            ) = match
            _apply_identity_vote(
                slot=track_slots[matched_track_id],
                detection=matched_detection,
                person_box=matched_person_box,
                face=face,
                face_box=face_box,
                frame_id=frame_id,
                timestamp=timestamp,
                identity_id=identity_id,
                source=source,
                identity_confidence=identity_confidence,
                association_confidence=association_confidence,
            )

    normalized_tracks = _normalize_track_slots(
        track_slots,
        ambiguity_margin=resolved_ambiguity_margin,
    )
    _link_unresolved_tracks_by_continuity(
        normalized_tracks,
        ambiguity_margin=resolved_ambiguity_margin,
    )
    fused_groups = _build_fused_groups(normalized_tracks)
    fused_tracks, track_lookup = _build_fused_tracks(
        fused_groups=fused_groups,
        job_id=job_id,
    )
    _annotate_fused_person_tracks(
        ordered_frames=ordered_frames,
        track_lookup=track_lookup,
    )

    fused_tracks.sort(
        key=lambda item: (
            int(item["frame_span"]["first_frame_id"] or 0),
            str(item["identity_id"] or ""),
            str(item["person_track_id"]),
        )
    )
    return {
        "enabled": True,
        "method": "object_face_fusion_v1",
        "tracks": fused_tracks,
    }
