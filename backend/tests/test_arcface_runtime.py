"""Unit tests for ArcFace runtime provider selection and fallback behavior."""

from types import SimpleNamespace

import numpy as np
import pytest

from app import face_identity


def _sample_image() -> np.ndarray:
    return np.full((64, 64, 3), fill_value=127, dtype=np.uint8)


def test_arcface_prefers_coreml_provider_when_available(monkeypatch):
    class _FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    class _FakeFaceAnalysis:
        def __init__(self, *, name, providers, root=None):
            self.name = name
            self.providers = providers
            self.root = root

        def prepare(self, ctx_id=0, det_size=(640, 640)):
            del ctx_id, det_size

        def get(self, image):
            del image
            return [SimpleNamespace(normed_embedding=np.array([3.0, 4.0], dtype=np.float32))]

    monkeypatch.setattr(face_identity, "_onnxruntime", _FakeOrt)
    monkeypatch.setattr(face_identity, "_InsightFaceAnalysis", _FakeFaceAnalysis)

    embedder = face_identity.ArcFaceRuntimeEmbedder(
        model_name="buffalo_l",
        provider_order=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        fallback_behavior="cpu",
        embedding_dimension=4,
    )

    assert embedder.runtime_backend == "arcface"
    assert embedder.active_provider == "CoreMLExecutionProvider"
    assert embedder.active_provider_chain == (
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )

    embedding = embedder.embed(_sample_image(), [8, 8, 40, 40])

    assert embedding.shape == (16,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_arcface_falls_back_to_cpu_when_coreml_init_fails(monkeypatch):
    class _FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    class _FakeFaceAnalysis:
        def __init__(self, *, name, providers, root=None):
            del name, root
            if providers and providers[0] == "CoreMLExecutionProvider":
                raise RuntimeError("coreml unavailable")
            self.providers = providers

        def prepare(self, ctx_id=0, det_size=(640, 640)):
            del ctx_id, det_size

        def get(self, image):
            del image
            return [SimpleNamespace(embedding=np.array([1.0, 0.0], dtype=np.float32))]

    monkeypatch.setattr(face_identity, "_onnxruntime", _FakeOrt)
    monkeypatch.setattr(face_identity, "_InsightFaceAnalysis", _FakeFaceAnalysis)

    embedder = face_identity.ArcFaceRuntimeEmbedder(
        model_name="buffalo_l",
        provider_order=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        fallback_behavior="cpu",
        embedding_dimension=32,
    )

    assert embedder.runtime_backend == "arcface"
    assert embedder.active_provider == "CPUExecutionProvider"
    assert embedder.active_provider_chain == ("CPUExecutionProvider",)


def test_arcface_uses_deterministic_fallback_when_runtime_unavailable(monkeypatch):
    monkeypatch.setattr(face_identity, "_onnxruntime", None)
    monkeypatch.setattr(face_identity, "_InsightFaceAnalysis", None)

    embedder = face_identity.ArcFaceRuntimeEmbedder(
        model_name="buffalo_l",
        provider_order=("CoreMLExecutionProvider",),
        fallback_behavior="cpu",
        embedding_dimension=24,
    )

    first = embedder.embed(_sample_image(), [8, 8, 40, 40])
    second = embedder.embed(_sample_image(), [8, 8, 40, 40])

    assert embedder.runtime_backend == "deterministic_fallback"
    assert embedder.active_provider == "deterministic_fallback"
    assert np.allclose(first, second)
