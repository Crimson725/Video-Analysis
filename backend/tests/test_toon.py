"""Tests for app.toon conversion wrapper."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.toon import ToonConversionError, convert_json_to_toon


class TestConvertJsonToToon:
    def test_missing_converter_script_raises(self, monkeypatch):
        fake_script = Path("/tmp/does-not-exist/convert_toon.mjs")
        monkeypatch.setattr("app.toon._TOON_CONVERTER", fake_script)

        with pytest.raises(ToonConversionError, match="script not found"):
            convert_json_to_toon(b'{"frame_id": 1}')

    def test_non_zero_exit_raises(self, monkeypatch):
        script = Path(__file__)
        monkeypatch.setattr("app.toon._TOON_CONVERTER", script)

        result = SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom")
        monkeypatch.setattr("app.toon.subprocess.run", lambda *args, **kwargs: result)

        with pytest.raises(ToonConversionError, match="TOON conversion failed"):
            convert_json_to_toon(b'{"frame_id": 1}')

    def test_success_returns_stdout_bytes(self, monkeypatch):
        script = Path(__file__)
        monkeypatch.setattr("app.toon._TOON_CONVERTER", script)

        result = SimpleNamespace(returncode=0, stdout=b"TOON_OK", stderr=b"")
        monkeypatch.setattr("app.toon.subprocess.run", lambda *args, **kwargs: result)

        payload = convert_json_to_toon(b'{"frame_id": 1}')

        assert payload == b"TOON_OK"
