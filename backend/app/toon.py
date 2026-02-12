"""Helpers for converting JSON analysis payloads to TOON format."""

from __future__ import annotations

import errno
import os
import subprocess
import tempfile
import time
from pathlib import Path


class ToonConversionError(RuntimeError):
    """Raised when JSON-to-TOON conversion fails."""


_TOON_RUNTIME_DIR = Path(__file__).resolve().parent.parent / "scripts" / "toon_runtime"
_TOON_CONVERTER = _TOON_RUNTIME_DIR / "convert_toon.mjs"


def _is_transient_converter_error(details: str) -> bool:
    """Return whether converter output looks like a transient runtime issue."""
    lowered = details.lower()
    markers = (
        "eagain",
        "temporarily unavailable",
        "resource temporarily unavailable",
        "eintr",
    )
    return any(marker in lowered for marker in markers)


def convert_json_to_toon(
    json_payload: bytes,
    timeout_seconds: float = 10.0,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 0.2,
) -> bytes:
    """Convert UTF-8 JSON bytes to TOON bytes using the Node helper script."""
    if not _TOON_CONVERTER.is_file():
        raise ToonConversionError(f"TOON converter script not found: {_TOON_CONVERTER}")
    if max_attempts < 1:
        raise ToonConversionError("TOON conversion max_attempts must be >= 1")

    input_file_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as input_file:
            input_file.write(json_payload)
            input_file.flush()
            input_file_path = input_file.name
    except OSError as exc:
        raise ToonConversionError("Unable to prepare TOON converter input file") from exc

    command = ["node", str(_TOON_CONVERTER), "--input", str(input_file_path)]
    last_error: ToonConversionError | None = None

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                result = subprocess.run(
                    command,
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
                transient_os_error = exc.errno in {errno.EAGAIN, errno.EINTR}
                if transient_os_error and attempt < max_attempts:
                    time.sleep(retry_backoff_seconds * attempt)
                    continue
                raise ToonConversionError("Unable to execute TOON converter") from exc

            if result.returncode != 0:
                stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
                stdout = (result.stdout or b"").decode("utf-8", errors="replace").strip()
                details = stderr or stdout or "unknown converter error"
                last_error = ToonConversionError(f"TOON conversion failed: {details}")
                if attempt < max_attempts and _is_transient_converter_error(details):
                    time.sleep(retry_backoff_seconds * attempt)
                    continue
                raise last_error

            if not result.stdout:
                raise ToonConversionError("TOON conversion produced empty output")
            return bytes(result.stdout)

        if last_error is not None:
            raise last_error
        raise ToonConversionError("TOON conversion failed after retries")
    finally:
        if input_file_path is not None:
            try:
                os.unlink(input_file_path)
            except OSError:
                pass
