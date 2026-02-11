"""Helpers for converting JSON analysis payloads to TOON format."""

from __future__ import annotations

import subprocess
from pathlib import Path


class ToonConversionError(RuntimeError):
    """Raised when JSON-to-TOON conversion fails."""


_TOON_RUNTIME_DIR = Path(__file__).resolve().parent.parent / "scripts" / "toon_runtime"
_TOON_CONVERTER = _TOON_RUNTIME_DIR / "convert_toon.mjs"


def convert_json_to_toon(json_payload: bytes, timeout_seconds: float = 10.0) -> bytes:
    """Convert UTF-8 JSON bytes to TOON bytes using the Node helper script."""
    if not _TOON_CONVERTER.is_file():
        raise ToonConversionError(f"TOON converter script not found: {_TOON_CONVERTER}")

    command = ["node", str(_TOON_CONVERTER)]
    try:
        result = subprocess.run(
            command,
            input=json_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_TOON_RUNTIME_DIR),
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise ToonConversionError("Node.js runtime not found in PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToonConversionError("TOON conversion timed out") from exc
    except OSError as exc:
        raise ToonConversionError("Unable to execute TOON converter") from exc

    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        stdout = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        details = stderr or stdout or "unknown converter error"
        raise ToonConversionError(f"TOON conversion failed: {details}")

    if not result.stdout:
        raise ToonConversionError("TOON conversion produced empty output")
    return bytes(result.stdout)
