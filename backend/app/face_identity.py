"""Face identity embedding and aggregation utilities."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)


def _normalized_vector(values: np.ndarray) -> np.ndarray:
    """Return an L2-normalized embedding vector."""
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        return values
    return values / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity for already-normalized vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


@dataclass(slots=True)
class FaceObservation:
    """One face observation with embedding-ready metadata."""

    scene_id: int
    frame_id: int
    timestamp: str
    face_id: int
    coordinates: list[int]
    confidence: float
    embedding: np.ndarray
    source: str

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.scene_id, self.frame_id, self.face_id)


@dataclass(slots=True)
class SceneCluster:
    """Scene-local identity cluster."""

    scene_person_id: str
    scene_id: int
    centroid: np.ndarray
    count: int
    observations: list[tuple[int, int]] = field(default_factory=list)

    def update(self, embedding: np.ndarray, frame_id: int, face_id: int) -> None:
        total = self.count + 1
        self.centroid = _normalized_vector((self.centroid * self.count + embedding) / total)
        self.count = total
        self.observations.append((frame_id, face_id))


@dataclass(slots=True)
class VideoCluster:
    """Video-global identity cluster."""

    video_person_id: str
    centroid: np.ndarray
    count: int
    scene_person_ids: list[str] = field(default_factory=list)

    def update(self, embedding: np.ndarray, scene_person_id: str) -> None:
        total = self.count + 1
        self.centroid = _normalized_vector((self.centroid * self.count + embedding) / total)
        self.count = total
        if scene_person_id not in self.scene_person_ids:
            self.scene_person_ids.append(scene_person_id)


class EdgeFaceTorchEmbedder:
    """PyTorch-backed EdgeFace-style embedding extractor with deterministic fallback."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_id: str,
        embedding_dimension: int,
        weights_path: str = "",
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.embedding_dimension = max(16, int(embedding_dimension))
        self._model: InceptionResnetV1 | None = None
        self._load_model(weights_path)

    def _load_model(self, weights_path: str) -> None:
        model = InceptionResnetV1(classify=False, pretrained=None).to(self.device).eval()
        candidate = Path(weights_path) if weights_path else None
        if candidate and candidate.is_file():
            try:
                state = torch.load(candidate, map_location=self.device)
                if isinstance(state, dict):
                    model.load_state_dict(state, strict=False)
                    logger.info(
                        "Loaded face identity weights from %s using model_id=%s",
                        candidate,
                        self.model_id,
                    )
            except Exception as exc:  # pragma: no cover - depends on runtime weights
                logger.warning("Failed loading face identity weights from %s: %s", candidate, exc)
        self._model = model

    def _fallback_embedding(self, crop_rgb: np.ndarray) -> np.ndarray:
        digest = hashlib.sha1(crop_rgb.tobytes()).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.embedding_dimension).astype(np.float32)
        return _normalized_vector(vector)

    def embed(self, image_bgr: np.ndarray, bbox: list[int]) -> np.ndarray:
        """Extract one normalized embedding for a face crop."""
        if image_bgr.size == 0 or len(bbox) != 4:
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        height, width = image_bgr.shape[:2]
        x1 = max(0, min(width - 1, int(bbox[0])))
        y1 = max(0, min(height - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(width, int(bbox[2])))
        y2 = max(y1 + 1, min(height, int(bbox[3])))
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self._model is None:  # pragma: no cover - guarded by constructor
            return self._fallback_embedding(crop_rgb)

        try:
            resized = cv2.resize(crop_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            # Facenet-style normalization.
            tensor = (tensor / 255.0 - 0.5) / 0.5
            with torch.no_grad():
                embedding = self._model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
            return _normalized_vector(embedding)
        except Exception as exc:
            logger.warning("Face embedding inference failed; using deterministic fallback: %s", exc)
            return self._fallback_embedding(crop_rgb)


def _best_similarity(
    embedding: np.ndarray,
    clusters: list[SceneCluster | VideoCluster],
) -> tuple[int | None, float, float]:
    """Return (best_idx, best_sim, second_best_sim)."""
    if not clusters:
        return None, 0.0, 0.0
    sims = [cosine_similarity(embedding, cluster.centroid) for cluster in clusters]
    ordered = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)
    best_idx, best_sim = ordered[0]
    second_best = ordered[1][1] if len(ordered) > 1 else 0.0
    return best_idx, float(best_sim), float(second_best)


def aggregate_scene_identities(
    observations: list[FaceObservation],
    *,
    similarity_threshold: float,
    ambiguity_margin: float,
) -> tuple[
    dict[tuple[int, int, int], dict[str, Any]],
    dict[int, list[SceneCluster]],
]:
    """Cluster observations into scene-local identities."""
    assignments: dict[tuple[int, int, int], dict[str, Any]] = {}
    clusters_by_scene: dict[int, list[SceneCluster]] = {}

    ordered = sorted(observations, key=lambda item: (item.scene_id, item.frame_id, item.face_id))
    for obs in ordered:
        clusters = clusters_by_scene.setdefault(obs.scene_id, [])
        best_idx, best_sim, second_best = _best_similarity(obs.embedding, clusters)
        ambiguous = (
            best_idx is not None
            and best_sim >= similarity_threshold
            and (best_sim - second_best) <= ambiguity_margin
        )

        if best_idx is not None and best_sim >= similarity_threshold and not ambiguous:
            cluster = clusters[best_idx]
            cluster.update(obs.embedding, obs.frame_id, obs.face_id)
            scene_person_id = cluster.scene_person_id
            match_confidence = best_sim
        else:
            scene_person_id = f"scene_{obs.scene_id}_person_{len(clusters) + 1}"
            cluster = SceneCluster(
                scene_person_id=scene_person_id,
                scene_id=obs.scene_id,
                centroid=obs.embedding.copy(),
                count=1,
                observations=[(obs.frame_id, obs.face_id)],
            )
            clusters.append(cluster)
            match_confidence = 1.0

        assignments[obs.key] = {
            "scene_person_id": scene_person_id,
            "match_confidence": float(match_confidence),
            "is_identity_ambiguous": bool(ambiguous),
        }
    return assignments, clusters_by_scene


def stitch_video_identities(
    clusters_by_scene: dict[int, list[SceneCluster]],
    *,
    similarity_threshold: float,
    ambiguity_margin: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Map scene-local identities into global video identities."""
    scene_to_video: dict[str, dict[str, Any]] = {}
    video_clusters: list[VideoCluster] = []
    next_video_id = 1

    scene_ids = sorted(clusters_by_scene.keys())
    for scene_id in scene_ids:
        for cluster in sorted(
            clusters_by_scene[scene_id],
            key=lambda item: item.scene_person_id,
        ):
            best_idx, best_sim, second_best = _best_similarity(cluster.centroid, video_clusters)
            ambiguous = (
                best_idx is not None
                and best_sim >= similarity_threshold
                and (best_sim - second_best) <= ambiguity_margin
            )

            if best_idx is not None and best_sim >= similarity_threshold and not ambiguous:
                video_cluster = video_clusters[best_idx]
                video_cluster.update(cluster.centroid, cluster.scene_person_id)
                video_person_id = video_cluster.video_person_id
                confidence = best_sim
            else:
                video_person_id = f"video_person_{next_video_id}"
                next_video_id += 1
                video_cluster = VideoCluster(
                    video_person_id=video_person_id,
                    centroid=cluster.centroid.copy(),
                    count=1,
                    scene_person_ids=[cluster.scene_person_id],
                )
                video_clusters.append(video_cluster)
                confidence = 1.0

            scene_to_video[cluster.scene_person_id] = {
                "video_person_id": video_person_id,
                "confidence": float(confidence),
                "is_ambiguous": bool(ambiguous),
            }

    video_summary = [
        {
            "video_person_id": cluster.video_person_id,
            "scene_person_ids": sorted(cluster.scene_person_ids),
            "cluster_size": cluster.count,
        }
        for cluster in sorted(video_clusters, key=lambda item: item.video_person_id)
    ]
    return scene_to_video, video_summary
