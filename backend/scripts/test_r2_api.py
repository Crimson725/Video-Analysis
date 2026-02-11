"""Smoke-test Cloudflare R2 credentials against the real S3-compatible API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[key] = value


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parents[1] / ".env.r2"),
        help="Path to env file with R2 credentials",
    )
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    try:
        account_id = required_env("R2_ACCOUNT_ID")
        bucket = required_env("R2_BUCKET")
        access_key_id = required_env("R2_ACCESS_KEY_ID")
        secret_access_key = required_env("R2_SECRET_ACCESS_KEY")
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2

    endpoint = os.getenv("R2_S3_ENDPOINT", "").strip() or (
        f"https://{account_id}.r2.cloudflarestorage.com"
    )

    print(f"Using endpoint: {endpoint}")
    print(f"Testing bucket: {bucket}")

    client = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    try:
        client.head_bucket(Bucket=bucket)
        print("head_bucket OK")
    except Exception as exc:  # pragma: no cover - network/auth runtime path
        print(f"R2 API test failed: {exc}")
        return 1

    print("R2 API smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
