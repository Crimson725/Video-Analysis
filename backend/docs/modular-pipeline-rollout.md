# Modular Pipeline Rollout

This backend now runs video processing through a modular stage registry and executor by default.

## Rollout Steps

1. Deploy with default `USE_LEGACY_ORCHESTRATION=false`.
2. Monitor `GET /status/{job_id}` responses for `current_stage`/`failed_stage` signals.
3. Verify `GET /results/{job_id}` includes top-level `pipeline` metadata and ordered `frames`.
4. Compare modular output against baseline jobs for parity in core frame fields:
   - `frame_id`
   - `timestamp`
   - `files`
   - `analysis`
   - `analysis_artifacts`
   - `metadata`
5. Keep regression tests green (`uv run pytest`) before and after each release.

## Rollback Switch

- Set `USE_LEGACY_ORCHESTRATION=true` to route background jobs through the deprecated monolithic path (`process_video_legacy`) while preserving API endpoints.
- Keep the flag only as a temporary safety valve; remove after sustained modular stability.
