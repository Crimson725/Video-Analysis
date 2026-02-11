# Cloudflare R2 Lifecycle Setup

Apply lifecycle policies per environment (dev/stage/prod) to manage object retention.

## What to configure

- Abort incomplete multipart uploads under `jobs/` after `R2_ABORT_MULTIPART_DAYS`
- Expire objects under `jobs/` after `R2_RETENTION_DAYS`

## Example configuration payload

Start from:

- `/Users/crimson2049/Video Analysis/backend/docs/r2-lifecycle.example.json`

Update:

- `Bucket` to your environment bucket
- `DaysAfterInitiation` and `Expiration.Days` to match environment policy

## Apply with AWS-compatible tooling

Configure your S3 client endpoint to:

- `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`

Then apply the lifecycle payload with your preferred S3-compatible tool or SDK.
