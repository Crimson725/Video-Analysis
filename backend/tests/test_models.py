"""Unit tests for PyTorch device/runtime helpers in app.models."""

from types import SimpleNamespace
from unittest.mock import patch

from app.models import select_torch_device, tensorflow_runtime_note


def test_select_torch_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr("app.models.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "app.models.torch.backends.mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    device = select_torch_device("auto")

    assert device.type == "cuda"


def test_select_torch_device_falls_back_to_mps_then_cpu(monkeypatch):
    monkeypatch.setattr("app.models.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "app.models.torch.backends.mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    assert select_torch_device("auto").type == "mps"
    assert select_torch_device("cpu").type == "cpu"

    monkeypatch.setattr(
        "app.models.torch.backends.mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )
    assert select_torch_device("mps").type == "cpu"


def test_select_torch_device_honors_explicit_cuda_and_fallback(monkeypatch):
    monkeypatch.setattr("app.models.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "app.models.torch.backends.mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    assert select_torch_device("cuda").type == "cpu"

    monkeypatch.setattr("app.models.torch.cuda.is_available", lambda: True)
    assert select_torch_device("cuda").type == "cuda"


def test_tensorflow_runtime_note_does_not_require_tensorflow():
    with patch("app.models.importlib.util.find_spec", return_value=None):
        assert "not installed" in tensorflow_runtime_note().lower()

    with patch("app.models.importlib.util.find_spec", return_value=object()):
        assert "not required" in tensorflow_runtime_note().lower()
