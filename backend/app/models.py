"""Singleton model loader for YOLO and MTCNN models (PyTorch-only stack)."""

import logging

import torch
from facenet_pytorch import MTCNN
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton loader for ML models (YOLO + MTCNN, all PyTorch)."""

    _instance: "ModelLoader | None" = None

    @classmethod
    def get(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            logger.info("PyTorch using CUDA GPU acceleration")
        else:
            logger.warning("CUDA not available — running on CPU (slower)")

        self.detector = YOLO("yolo11n.pt")
        self.segmenter = YOLO("yolo11n-seg.pt")
        self.face_detector = MTCNN(keep_all=True, device=device)
