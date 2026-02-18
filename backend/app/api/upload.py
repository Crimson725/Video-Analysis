"""API-layer helpers for request-bound upload handling."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile


async def stream_upload_to_local_file(
    *,
    file: UploadFile,
    destination: Path,
    max_upload_bytes: int,
    cleanup_root: Path,
    job_id: str,
) -> None:
    """Stream upload body to deterministic local file with size guard."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_upload_bytes:
                    destination.unlink(missing_ok=True)
                    shutil.rmtree(cleanup_root / job_id, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"File exceeds {max_upload_bytes // (1024 * 1024)} MB limit",
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        shutil.rmtree(cleanup_root / job_id, ignore_errors=True)
        raise HTTPException(500, str(exc)) from exc
