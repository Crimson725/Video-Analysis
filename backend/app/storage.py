"""Media storage abstraction and Cloudflare R2 implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


FrameKind = Literal["original", "seg", "det", "face"]
AnalysisArtifactKind = Literal["json", "toon"]

_FRAME_PREFIX: dict[FrameKind, str] = {
    "original": "original",
    "seg": "seg",
    "det": "det",
    "face": "face",
}

_ANALYSIS_PREFIX: dict[AnalysisArtifactKind, str] = {
    "json": "json",
    "toon": "toon",
}

_ANALYSIS_EXTENSION: dict[AnalysisArtifactKind, str] = {
    "json": "json",
    "toon": "toon",
}

_ANALYSIS_CONTENT_TYPE: dict[AnalysisArtifactKind, str] = {
    "json": "application/json",
    "toon": "application/x-toon",
}


class MediaStoreError(RuntimeError):
    """Base exception for media storage failures."""


class MediaStoreConfigError(MediaStoreError):
    """Raised when media storage configuration is missing or invalid."""


class MediaStore(Protocol):
    """Storage interface used by the analysis pipeline."""

    def upload_source_video(
        self,
        job_id: str,
        file_path: str,
        content_type: str,
        source_extension: str | None = None,
    ) -> str:
        """Upload a source video and return the stored object key."""

    def upload_frame_image(self, job_id: str, frame_kind: FrameKind, frame_id: int, image_bytes: bytes) -> str:
        """Upload a generated frame artifact and return the stored object key."""

    def upload_analysis_artifact(
        self,
        job_id: str,
        artifact_kind: AnalysisArtifactKind,
        frame_id: int,
        payload: bytes,
    ) -> str:
        """Upload a generated analysis artifact and return the stored object key."""

    def read_object(self, object_key: str) -> bytes:
        """Read object bytes for a key."""

    def delete_object(self, object_key: str) -> None:
        """Delete an object key."""

    def verify_object(self, object_key: str) -> bool:
        """Verify an object exists in storage."""

    def sign_read_url(self, object_key: str, expires_in: int | None = None) -> str:
        """Generate a signed read URL for an object key."""


def _normalize_extension(source_extension: str | None, default: str = "mp4") -> str:
    """Normalize file extension to a lower-case token without leading dot."""
    if not source_extension:
        return default
    normalized = source_extension.strip().lower().lstrip(".")
    if not normalized:
        return default
    return "".join(ch for ch in normalized if ch.isalnum()) or default


def build_source_video_key(job_id: str, source_extension: str | None = "mp4") -> str:
    """Build the deterministic key for a source video object."""
    extension = _normalize_extension(source_extension, default="mp4")
    return f"jobs/{job_id}/input/source.{extension}"


def build_frame_key(job_id: str, frame_kind: FrameKind, frame_id: int) -> str:
    """Build deterministic key for a frame artifact object."""
    return f"jobs/{job_id}/frames/{_FRAME_PREFIX[frame_kind]}/frame_{frame_id}.jpg"


def build_analysis_key(job_id: str, artifact_kind: AnalysisArtifactKind, frame_id: int) -> str:
    """Build deterministic key for an analysis artifact object."""
    artifact_dir = _ANALYSIS_PREFIX[artifact_kind]
    artifact_ext = _ANALYSIS_EXTENSION[artifact_kind]
    return f"jobs/{job_id}/analysis/{artifact_dir}/frame_{frame_id}.{artifact_ext}"


def build_r2_endpoint(account_id: str) -> str:
    """Build the Cloudflare R2 S3-compatible endpoint URL."""
    return f"https://{account_id}.r2.cloudflarestorage.com"


@dataclass(slots=True)
class R2MediaStore:
    """Cloudflare R2 implementation of the MediaStore interface."""

    account_id: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    default_url_ttl_seconds: int = 3600
    s3_client: Any | None = None

    def __post_init__(self) -> None:
        missing = []
        if not self.account_id:
            missing.append("R2_ACCOUNT_ID")
        if not self.bucket:
            missing.append("R2_BUCKET")
        if not self.access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        if missing:
            fields = ", ".join(missing)
            raise MediaStoreConfigError(f"Missing required R2 configuration: {fields}")

        if self.default_url_ttl_seconds <= 0:
            raise MediaStoreConfigError("R2_URL_TTL_SECONDS must be greater than zero")

        if self.s3_client is None:
            self.s3_client = self._build_client()

    def _build_client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised in runtime only
            raise MediaStoreConfigError(
                "boto3 is not installed. Add boto3/botocore dependencies for R2 support."
            ) from exc

        return boto3.client(
            "s3",
            region_name="auto",
            endpoint_url=build_r2_endpoint(self.account_id),
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )

    def upload_source_video(
        self,
        job_id: str,
        file_path: str,
        content_type: str,
        source_extension: str | None = None,
    ) -> str:
        object_key = build_source_video_key(job_id, source_extension=source_extension)
        self._upload_file(file_path, object_key, content_type or "video/mp4")
        return object_key

    def upload_frame_image(self, job_id: str, frame_kind: FrameKind, frame_id: int, image_bytes: bytes) -> str:
        object_key = build_frame_key(job_id, frame_kind, frame_id)
        self._put_object(object_key, image_bytes, "image/jpeg")
        return object_key

    def upload_analysis_artifact(
        self,
        job_id: str,
        artifact_kind: AnalysisArtifactKind,
        frame_id: int,
        payload: bytes,
    ) -> str:
        object_key = build_analysis_key(job_id, artifact_kind, frame_id)
        self._put_object(object_key, payload, _ANALYSIS_CONTENT_TYPE[artifact_kind])
        return object_key

    def read_object(self, object_key: str) -> bytes:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=object_key)
            body = response["Body"].read()
            return bytes(body)
        except Exception as exc:  # pragma: no cover - depends on external SDK
            raise MediaStoreError(f"Failed to read object '{object_key}'") from exc

    def delete_object(self, object_key: str) -> None:
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:  # pragma: no cover - depends on external SDK
            raise MediaStoreError(f"Failed to delete object '{object_key}'") from exc

    def verify_object(self, object_key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except Exception:
            return False

    def sign_read_url(self, object_key: str, expires_in: int | None = None) -> str:
        ttl = expires_in or self.default_url_ttl_seconds
        if ttl <= 0:
            raise MediaStoreError("Signed URL expiration must be greater than zero")
        try:
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_key},
                ExpiresIn=ttl,
            )
        except Exception as exc:  # pragma: no cover - depends on external SDK
            raise MediaStoreError(f"Failed to generate signed URL for '{object_key}'") from exc

    def _upload_file(self, file_path: str, object_key: str, content_type: str) -> None:
        try:
            self.s3_client.upload_file(
                file_path,
                self.bucket,
                object_key,
                ExtraArgs={"ContentType": content_type},
            )
        except Exception as exc:  # pragma: no cover - depends on external SDK
            raise MediaStoreError(f"Failed to upload file '{file_path}' to '{object_key}'") from exc

    def _put_object(self, object_key: str, body: bytes, content_type: str) -> None:
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=body,
                ContentType=content_type,
            )
        except Exception as exc:  # pragma: no cover - depends on external SDK
            raise MediaStoreError(f"Failed to upload object '{object_key}'") from exc
