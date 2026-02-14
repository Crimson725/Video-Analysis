"""Singleton model loader for YOLO and MTCNN models (PyTorch-only stack)."""

import importlib.util
import logging
import os

import torch
from facenet_pytorch import MTCNN
from ultralytics import YOLO

from app.config import normalize_face_identity_model_id

logger = logging.getLogger(__name__)


def select_torch_device(preferred_backend: str = "auto") -> torch.device:
    """Resolve torch device with backend priority and graceful fallback."""
    backend = (preferred_backend or "auto").strip().lower()
    if backend not in {"auto", "cuda", "mps", "cpu"}:
        backend = "auto"

    if backend in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")

    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps is not None
        and torch.backends.mps.is_available()
    )
    if backend in {"auto", "mps"} and mps_available:
        return torch.device("mps")

    return torch.device("cpu")


def edgeface_runtime_note() -> str:
    """Return runtime note documenting EdgeFace uses a TensorFlow-free path."""
    if importlib.util.find_spec("tensorflow") is None:
        return "TensorFlow is not installed; EdgeFace identity runtime uses PyTorch only."
    return "TensorFlow is installed but not required; EdgeFace identity runtime uses PyTorch only."


class ModelLoader:
    """Singleton loader for ML models (YOLO + MTCNN, all PyTorch)."""

    _instance: "ModelLoader | None" = None

    @classmethod
    def get(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        preferred_backend = os.getenv("FACE_IDENTITY_BACKEND", "auto")
        model_id = normalize_face_identity_model_id(os.getenv("FACE_IDENTITY_MODEL_ID"))
        device = select_torch_device(preferred_backend)
        self.device = device
        logger.info(
            "EdgeFace device selection model_profile=%s backend=%s resolved_device=%s",
            model_id,
            preferred_backend,
            device.type,
        )
        if device.type == "cuda":
            logger.info("PyTorch using CUDA GPU acceleration")
        elif device.type == "mps":
            logger.info("PyTorch using Apple Metal (MPS) acceleration")
        else:
            logger.warning("CUDA not available — running on CPU (slower)")

        self.detector = YOLO("yolo11n.pt")
        self.segmenter = YOLO("yolo11n-seg.pt")
        self.face_detector = MTCNN(keep_all=True, device=device)
