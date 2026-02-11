"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for API and media storage."""

    r2_account_id: str
    r2_bucket: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_url_ttl_seconds: int
    r2_retention_days: int
    r2_abort_multipart_days: int
    temp_media_dir: str
    cleanup_local_video_after_upload_default: bool

    @classmethod
    def from_env(cls) -> "Settings":
        default_temp = Path(__file__).resolve().parent.parent / "tmp_media"
        return cls(
            r2_account_id=os.getenv("R2_ACCOUNT_ID", "").strip(),
            r2_bucket=os.getenv("R2_BUCKET", "").strip(),
            r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            r2_url_ttl_seconds=_read_int("R2_URL_TTL_SECONDS", default=3600, minimum=1),
            r2_retention_days=_read_int("R2_RETENTION_DAYS", default=7, minimum=1),
            r2_abort_multipart_days=_read_int("R2_ABORT_MULTIPART_DAYS", default=1, minimum=1),
            temp_media_dir=os.getenv("TEMP_MEDIA_DIR", str(default_temp)).strip(),
            cleanup_local_video_after_upload_default=_read_bool(
                "CLEANUP_LOCAL_VIDEO_AFTER_UPLOAD_DEFAULT",
                default=True,
            ),
        )

    def missing_r2_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.r2_account_id:
            missing.append("R2_ACCOUNT_ID")
        if not self.r2_bucket:
            missing.append("R2_BUCKET")
        if not self.r2_access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.r2_secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        return missing
