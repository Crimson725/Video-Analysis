# Modular Pipeline Rollout

This backend now runs video processing through a modular stage registry and executor by default.

## Rollout Steps

1. Deploy with modular orchestration enabled (default behavior).
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

## Rollback

- No alternate orchestration path is maintained. Fixes should be made in the modular pipeline stages/registry/executor directly.
