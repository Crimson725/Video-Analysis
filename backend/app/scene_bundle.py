"""Canonical SceneBundle builder: keyframe selection, derived features, overlay generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FrameData:
    """Data for a single selected keyframe within a SceneBundle."""

    frame_id: int
    timestamp_sec: float
    original_image_key: str
    overlay_image_key: str = ""
    original_url: str = ""
    overlay_url: str = ""
    face_crop_urls: list[str] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    segmentation: list[dict[str, Any]] = field(default_factory=list)
    faces: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TrackMeta:
    """Track metadata entry in the tracks_index."""

    track_id: str
    label: str
    entity_type: str  # "person" or "object"
    confidence_mean: float = 0.0
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    frame_ids: list[int] = field(default_factory=list)
    bboxes: dict[int, list[int]] = field(default_factory=dict)
    video_person_id: str | None = None


@dataclass(slots=True)
class DerivedFrameFeatures:
    """Numeric features computed for one frame."""

    frame_id: int
    centroids: dict[str, tuple[float, float]] = field(default_factory=dict)
    bbox_area_pct: dict[str, float] = field(default_factory=dict)
    polygon_area_pct: dict[str, float] = field(default_factory=dict)
    pairwise_distances: list[dict[str, Any]] = field(default_factory=list)
    pairwise_iou: list[dict[str, Any]] = field(default_factory=list)
    spatial_ordering: list[dict[str, str]] = field(default_factory=list)
    near_edges: list[dict[str, str]] = field(default_factory=list)
    velocities: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SceneBundle:
    """Canonical scene data object for the KG enrichment pipeline."""

    scene_id: str
    job_id: str
    source_key: str
    scene_time_span: tuple[float, float]
    frames: list[FrameData] = field(default_factory=list)
    tracks_index: dict[str, TrackMeta] = field(default_factory=dict)
    derived_features: list[DerivedFrameFeatures] = field(default_factory=list)
    selected_frame_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------

def _frame_timestamp(frame: dict[str, Any]) -> float:
    """Parse frame timestamp to float seconds."""
    ts = frame.get("timestamp", "")
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = str(ts).split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    try:
        return float(ts)
    except (ValueError, TypeError):
        return 0.0


def _centroid_from_bbox(bbox: list[int | float]) -> tuple[float, float]:
    """Compute centroid from [x1, y1, x2, y2] bbox."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _extract_track_id(det: dict[str, Any]) -> str:
    """Get the best available track ID from a detection."""
    return (
        det.get("object_track_id")
        or det.get("person_track_id")
        or det.get("track_id")
        or ""
    )


def _extract_face_id(face: dict[str, Any]) -> str:
    """Get the best available person identity from a face detection."""
    return (
        face.get("video_person_id")
        or face.get("scene_person_id")
        or face.get("identity_id")
        or ""
    )


def _build_frame_track_map(
    scene_frames: list[dict[str, Any]],
) -> dict[int, dict[str, tuple[float, float]]]:
    """Map frame_id -> {track_id -> centroid} from detection data."""
    result: dict[int, dict[str, tuple[float, float]]] = {}
    for frame in scene_frames:
        fid = int(frame.get("frame_id", 0))
        analysis = frame.get("analysis", {})
        centroids: dict[str, tuple[float, float]] = {}
        for det in analysis.get("object_detection", []):
            tid = _extract_track_id(det)
            if tid and det.get("box"):
                centroids[tid] = _centroid_from_bbox(det["box"])
        for face in analysis.get("face_recognition", []):
            fid_name = _extract_face_id(face)
            if fid_name and face.get("coordinates"):
                centroids[fid_name] = _centroid_from_bbox(face["coordinates"])
        result[fid] = centroids
    return result


def select_keyframes(
    scene_frames: list[dict[str, Any]],
    *,
    max_keyframes: int = 12,
    motion_threshold: float = 80.0,
    interaction_distance_threshold: float = 150.0,
) -> list[int]:
    """Select keyframes using semantic-change-driven rules.

    Always includes first and last frame. Additional frames are triggered by:
    - New track appearance/disappearance
    - Motion above threshold
    - Interaction proximity (pairwise distance drops below threshold)
    - Face identity change
    """
    if not scene_frames:
        return []

    sorted_frames = sorted(scene_frames, key=lambda f: (
        _frame_timestamp(f), int(f.get("frame_id", 0))
    ))
    frame_ids = [int(f.get("frame_id", 0)) for f in sorted_frames]

    if len(frame_ids) <= 2:
        return frame_ids

    # Always include first and last
    selected: set[int] = {frame_ids[0], frame_ids[-1]}
    # Priority scoring: higher score = more semantically important
    scores: dict[int, float] = {fid: 0.0 for fid in frame_ids}

    track_map = _build_frame_track_map(sorted_frames)
    prev_tracks: set[str] | None = None
    prev_centroids: dict[str, tuple[float, float]] = {}
    prev_face_ids: set[str] = set()

    for frame in sorted_frames:
        fid = int(frame.get("frame_id", 0))
        analysis = frame.get("analysis", {})
        current_tracks = set(track_map.get(fid, {}).keys())
        current_centroids = track_map.get(fid, {})

        # Track appearance/disappearance
        if prev_tracks is not None:
            new_tracks = current_tracks - prev_tracks
            gone_tracks = prev_tracks - current_tracks
            if new_tracks or gone_tracks:
                selected.add(fid)
                scores[fid] += 3.0  # High priority for track events

        # Motion threshold
        for tid, centroid in current_centroids.items():
            if tid in prev_centroids:
                dist = _euclidean(centroid, prev_centroids[tid])
                if dist > motion_threshold:
                    scores[fid] = max(scores.get(fid, 0.0), 2.0)

        # Interaction proximity
        track_ids_list = list(current_centroids.keys())
        for i in range(len(track_ids_list)):
            for j in range(i + 1, len(track_ids_list)):
                d = _euclidean(
                    current_centroids[track_ids_list[i]],
                    current_centroids[track_ids_list[j]],
                )
                if d < interaction_distance_threshold:
                    scores[fid] = max(scores.get(fid, 0.0), 2.5)

        # Face identity change
        current_faces: set[str] = set()
        for face in analysis.get("face_recognition", []):
            fid_name = _extract_face_id(face)
            if fid_name:
                current_faces.add(fid_name)
        if prev_face_ids and current_faces != prev_face_ids:
            scores[fid] = max(scores.get(fid, 0.0), 2.0)

        prev_tracks = current_tracks
        prev_centroids = current_centroids
        prev_face_ids = current_faces

    # Add high-score frames
    remaining_budget = max_keyframes - len(selected)
    if remaining_budget > 0:
        scored_candidates = [
            (fid, scores.get(fid, 0.0)) for fid in frame_ids if fid not in selected
        ]
        scored_candidates.sort(key=lambda x: -x[1])
        for fid, score in scored_candidates[:remaining_budget]:
            if score > 0:
                selected.add(fid)

    # If still under budget, spread temporally
    remaining_budget = max_keyframes - len(selected)
    if remaining_budget > 0:
        unselected = [fid for fid in frame_ids if fid not in selected]
        if unselected:
            step = max(1, len(unselected) // (remaining_budget + 1))
            for idx in range(0, len(unselected), step):
                if len(selected) >= max_keyframes:
                    break
                selected.add(unselected[idx])

    # Cap and sort
    result = sorted(selected)
    if len(result) > max_keyframes:
        # Keep first, last, plus highest-scoring
        first_last = {result[0], result[-1]}
        middle = [(fid, scores.get(fid, 0.0)) for fid in result if fid not in first_last]
        middle.sort(key=lambda x: -x[1])
        keep = first_last | {fid for fid, _ in middle[: max_keyframes - 2]}
        result = sorted(keep)

    return result


# ---------------------------------------------------------------------------
# Derived numeric features
# ---------------------------------------------------------------------------

def compute_bbox_centroid(bbox: list[int | float]) -> tuple[float, float]:
    """Compute centroid from [x1, y1, x2, y2]."""
    return _centroid_from_bbox(bbox)


def compute_bbox_area_pct(bbox: list[int | float], frame_width: int, frame_height: int) -> float:
    """Compute bbox area as percentage of frame area."""
    if frame_width <= 0 or frame_height <= 0:
        return 0.0
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    area = abs(x2 - x1) * abs(y2 - y1)
    return (area / (frame_width * frame_height)) * 100.0


def compute_polygon_area(polygon: list[list[int | float]]) -> float:
    """Compute polygon area using the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def compute_polygon_area_pct(
    polygon: list[list[int | float]],
    frame_width: int,
    frame_height: int,
) -> float:
    """Compute polygon area as percentage of frame area."""
    if frame_width <= 0 or frame_height <= 0:
        return 0.0
    return (compute_polygon_area(polygon) / (frame_width * frame_height)) * 100.0


def compute_iou(bbox_a: list[int | float], bbox_b: list[int | float]) -> float:
    """Compute intersection-over-union for two [x1,y1,x2,y2] bounding boxes."""
    x1 = max(float(bbox_a[0]), float(bbox_b[0]))
    y1 = max(float(bbox_a[1]), float(bbox_b[1]))
    x2 = min(float(bbox_a[2]), float(bbox_b[2]))
    y2 = min(float(bbox_a[3]), float(bbox_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = abs(float(bbox_a[2]) - float(bbox_a[0])) * abs(float(bbox_a[3]) - float(bbox_a[1]))
    area_b = abs(float(bbox_b[2]) - float(bbox_b[0])) * abs(float(bbox_b[3]) - float(bbox_b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_spatial_ordering(
    id_a: str,
    centroid_a: tuple[float, float],
    id_b: str,
    centroid_b: tuple[float, float],
) -> list[dict[str, str]]:
    """Compute spatial ordering relations between two entities."""
    relations: list[dict[str, str]] = []
    if centroid_a[0] < centroid_b[0]:
        relations.append({"subject": id_a, "predicate": "left_of", "object": id_b})
    elif centroid_a[0] > centroid_b[0]:
        relations.append({"subject": id_a, "predicate": "right_of", "object": id_b})
    if centroid_a[1] < centroid_b[1]:
        relations.append({"subject": id_a, "predicate": "above", "object": id_b})
    elif centroid_a[1] > centroid_b[1]:
        relations.append({"subject": id_a, "predicate": "below", "object": id_b})
    return relations


def compute_derived_features_for_frame(
    frame: dict[str, Any],
    *,
    frame_width: int = 1920,
    frame_height: int = 1080,
    near_threshold: float = 150.0,
    prev_centroids: dict[str, tuple[float, float]] | None = None,
    prev_timestamp: float | None = None,
) -> DerivedFrameFeatures:
    """Compute all derived numeric features for a single frame."""
    fid = int(frame.get("frame_id", 0))
    analysis = frame.get("analysis", {})
    features = DerivedFrameFeatures(frame_id=fid)
    timestamp = _frame_timestamp(frame)

    # Collect track bboxes and centroids
    track_bboxes: dict[str, list[int | float]] = {}
    for det in analysis.get("object_detection", []):
        tid = _extract_track_id(det)
        if tid and det.get("box"):
            track_bboxes[tid] = det["box"]
            features.centroids[tid] = _centroid_from_bbox(det["box"])
            features.bbox_area_pct[tid] = compute_bbox_area_pct(
                det["box"], frame_width, frame_height
            )
    for face in analysis.get("face_recognition", []):
        fid_name = _extract_face_id(face)
        if fid_name and face.get("coordinates"):
            track_bboxes[fid_name] = face["coordinates"]
            features.centroids[fid_name] = _centroid_from_bbox(face["coordinates"])
            features.bbox_area_pct[fid_name] = compute_bbox_area_pct(
                face["coordinates"], frame_width, frame_height
            )

    # Polygon areas from segmentation
    for seg in analysis.get("semantic_segmentation", []):
        polygon = seg.get("mask_polygon", [])
        class_name = seg.get("class", seg.get("class_name", "unknown"))
        seg_id = f"seg_{class_name}_{seg.get('object_id', 0)}"
        if polygon:
            features.polygon_area_pct[seg_id] = compute_polygon_area_pct(
                polygon, frame_width, frame_height
            )

    # Pairwise computations
    track_ids = list(features.centroids.keys())
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            id_a, id_b = track_ids[i], track_ids[j]
            c_a, c_b = features.centroids[id_a], features.centroids[id_b]
            dist = _euclidean(c_a, c_b)

            features.pairwise_distances.append({
                "track_a": id_a,
                "track_b": id_b,
                "distance": round(dist, 2),
            })

            # IoU if both have bboxes
            if id_a in track_bboxes and id_b in track_bboxes:
                iou = compute_iou(track_bboxes[id_a], track_bboxes[id_b])
                features.pairwise_iou.append({
                    "track_a": id_a,
                    "track_b": id_b,
                    "iou": round(iou, 4),
                })

            # Spatial ordering
            features.spatial_ordering.extend(
                compute_spatial_ordering(id_a, c_a, id_b, c_b)
            )

            # Near edges
            if dist < near_threshold:
                features.near_edges.append({
                    "track_a": id_a,
                    "track_b": id_b,
                })

    # Velocities across frames
    if prev_centroids is not None and prev_timestamp is not None:
        dt = timestamp - prev_timestamp
        if dt > 0:
            for tid, centroid in features.centroids.items():
                if tid in prev_centroids:
                    displacement = _euclidean(centroid, prev_centroids[tid])
                    features.velocities[tid] = round(displacement / dt, 2)

    return features


# ---------------------------------------------------------------------------
# Overlay image generation
# ---------------------------------------------------------------------------

def generate_overlay_image(
    original_image_bytes: bytes,
    frame_data: dict[str, Any],
) -> bytes:
    """Draw bboxes, track IDs, face IDs, and polygon outlines on a frame image.

    Returns PNG bytes of the annotated overlay image.
    """
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(BytesIO(original_image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

    analysis = frame_data.get("analysis", {})

    # Color palette for tracks
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    color_idx = 0

    # Draw detection bboxes and track ID labels
    for det in analysis.get("object_detection", []):
        box = det.get("box")
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = colors[color_idx % len(colors)]
        color_idx += 1

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        track_id = _extract_track_id(det)
        label = det.get("label", "")
        label_text = f"{track_id}" if track_id else label
        if label_text:
            # Draw label background
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)

    # Draw face bboxes and person ID labels
    for face in analysis.get("face_recognition", []):
        coords = face.get("coordinates")
        if not coords or len(coords) < 4:
            continue
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        color = (0, 200, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        person_id = _extract_face_id(face)
        if person_id:
            bbox_text = draw.textbbox((0, 0), person_id, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            draw.rectangle([x1, y2, x1 + text_w + 4, y2 + text_h + 4], fill=color)
            draw.text((x1 + 2, y2 + 2), person_id, fill=(255, 255, 255), font=font)

    # Draw segmentation polygon outlines
    for seg in analysis.get("semantic_segmentation", []):
        polygon = seg.get("mask_polygon", [])
        if polygon and len(polygon) >= 3:
            flat_points = [(int(pt[0]), int(pt[1])) for pt in polygon if len(pt) >= 2]
            if len(flat_points) >= 3:
                color = colors[color_idx % len(colors)]
                color_idx += 1
                draw.polygon(flat_points, outline=color)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Track index builder
# ---------------------------------------------------------------------------

def build_tracks_index(
    scene_frames: list[dict[str, Any]],
    *,
    job_id: str,
    scene_id: str,
) -> dict[str, TrackMeta]:
    """Build tracks_index mapping from scene CV data."""
    tracks: dict[str, TrackMeta] = {}

    for frame in scene_frames:
        fid = int(frame.get("frame_id", 0))
        analysis = frame.get("analysis", {})

        for det in analysis.get("object_detection", []):
            tid = _extract_track_id(det)
            if not tid:
                continue
            if tid not in tracks:
                tracks[tid] = TrackMeta(
                    track_id=tid,
                    label=det.get("label", "unknown"),
                    entity_type="object",
                    confidence_mean=0.0,
                    first_frame_id=fid,
                    last_frame_id=fid,
                    frame_ids=[],
                    bboxes={},
                )
            meta = tracks[tid]
            meta.frame_ids.append(fid)
            if det.get("box"):
                meta.bboxes[fid] = det["box"]
            if meta.first_frame_id is None or fid < meta.first_frame_id:
                meta.first_frame_id = fid
            if meta.last_frame_id is None or fid > meta.last_frame_id:
                meta.last_frame_id = fid
            # Update person-related fields
            person_id = det.get("person_identity_id") or det.get("person_track_id")
            if person_id:
                meta.video_person_id = person_id
                meta.entity_type = "person"

        for face in analysis.get("face_recognition", []):
            person_id = _extract_face_id(face)
            if not person_id:
                continue
            if person_id not in tracks:
                tracks[person_id] = TrackMeta(
                    track_id=person_id,
                    label="person",
                    entity_type="person",
                    confidence_mean=0.0,
                    first_frame_id=fid,
                    last_frame_id=fid,
                    frame_ids=[],
                    bboxes={},
                    video_person_id=person_id,
                )
            meta = tracks[person_id]
            meta.frame_ids.append(fid)
            if face.get("coordinates"):
                meta.bboxes[fid] = face["coordinates"]
            if meta.first_frame_id is None or fid < meta.first_frame_id:
                meta.first_frame_id = fid
            if meta.last_frame_id is None or fid > meta.last_frame_id:
                meta.last_frame_id = fid

    # Compute mean confidences
    for tid, meta in tracks.items():
        if meta.frame_ids:
            meta.confidence_mean = 1.0  # Default since we don't track per-frame conf here

    return tracks


# ---------------------------------------------------------------------------
# Full SceneBundle builder
# ---------------------------------------------------------------------------

def build_scene_bundle(
    *,
    job_id: str,
    scene_id: int,
    source_key: str,
    start_sec: float,
    end_sec: float,
    scene_frames: list[dict[str, Any]],
    media_store: Any,
    max_keyframes: int = 12,
    motion_threshold: float = 80.0,
    interaction_distance_threshold: float = 150.0,
    near_threshold: float = 150.0,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> SceneBundle:
    """Orchestrate full SceneBundle construction.

    load scene JSON -> select keyframes -> compute derived features
    -> generate overlays -> resolve signed URLs -> assemble SceneBundle
    """
    sid = str(scene_id)

    # Select keyframes
    selected_ids = select_keyframes(
        scene_frames,
        max_keyframes=max_keyframes,
        motion_threshold=motion_threshold,
        interaction_distance_threshold=interaction_distance_threshold,
    )

    # Build tracks index
    tracks_index = build_tracks_index(scene_frames, job_id=job_id, scene_id=sid)

    # Filter to selected frames only
    selected_frames_by_id: dict[int, dict[str, Any]] = {}
    for frame in scene_frames:
        fid = int(frame.get("frame_id", 0))
        if fid in selected_ids:
            selected_frames_by_id[fid] = frame

    # Compute derived features across selected frames
    derived_features: list[DerivedFrameFeatures] = []
    prev_centroids: dict[str, tuple[float, float]] | None = None
    prev_timestamp: float | None = None

    for fid in selected_ids:
        frame = selected_frames_by_id.get(fid)
        if frame is None:
            continue
        features = compute_derived_features_for_frame(
            frame,
            frame_width=frame_width,
            frame_height=frame_height,
            near_threshold=near_threshold,
            prev_centroids=prev_centroids,
            prev_timestamp=prev_timestamp,
        )
        derived_features.append(features)
        prev_centroids = features.centroids.copy()
        prev_timestamp = _frame_timestamp(frame)

    # Build frame data with overlays and URLs
    frames: list[FrameData] = []
    for fid in selected_ids:
        frame = selected_frames_by_id.get(fid)
        if frame is None:
            continue

        files = frame.get("files", {})
        analysis = frame.get("analysis", {})
        timestamp_sec = _frame_timestamp(frame)

        original_key = str(files.get("original", ""))

        # Generate overlay image
        overlay_key = ""
        if original_key and media_store is not None:
            try:
                original_bytes = media_store.read_object(original_key)
                overlay_bytes = generate_overlay_image(original_bytes, frame)
                # Upload overlay using a scene artifact pattern
                overlay_obj_key = f"jobs/{job_id}/scene/overlays/scene_{scene_id}_frame_{fid}.png"
                media_store._put_object(overlay_obj_key, overlay_bytes, "image/png")
                overlay_key = overlay_obj_key
            except Exception:
                overlay_key = ""

        # Resolve signed URLs
        original_url = ""
        overlay_url = ""
        if media_store is not None:
            try:
                if original_key:
                    original_url = media_store.sign_read_url(original_key)
            except Exception:
                pass
            try:
                if overlay_key:
                    overlay_url = media_store.sign_read_url(overlay_key)
            except Exception:
                pass

        # Collect face crop URLs
        face_crop_urls: list[str] = []
        for face in analysis.get("face_recognition", []):
            crop_key = face.get("crop_key", "")
            if crop_key and media_store is not None:
                try:
                    face_crop_urls.append(media_store.sign_read_url(crop_key))
                except Exception:
                    pass

        frame_data = FrameData(
            frame_id=fid,
            timestamp_sec=timestamp_sec,
            original_image_key=original_key,
            overlay_image_key=overlay_key,
            original_url=original_url,
            overlay_url=overlay_url,
            face_crop_urls=face_crop_urls,
            tracks=analysis.get("object_detection", []),
            segmentation=analysis.get("semantic_segmentation", []),
            faces=analysis.get("face_recognition", []),
        )
        frames.append(frame_data)

    return SceneBundle(
        scene_id=sid,
        job_id=job_id,
        source_key=source_key,
        scene_time_span=(start_sec, end_sec),
        frames=frames,
        tracks_index=tracks_index,
        derived_features=derived_features,
        selected_frame_ids=selected_ids,
    )
