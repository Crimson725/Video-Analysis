# Chunked Tracking Rollout

This backend supports feature-flagged rollout for concurrent branch execution and
video-level 3x3-zone chunk-tracking summaries.

## Branch Concurrency Controls

- `ENABLE_PIPELINE_BRANCH_CONCURRENCY=true|false`
  - Enables dependency-aware concurrent scheduling for independent branches.
- `PIPELINE_FRAME_BRANCH_WORKER_BUDGET=<int>`
  - Budget hint for frame-analysis branch scheduling.
- `PIPELINE_CHUNK_BRANCH_WORKER_BUDGET=<int>`
  - Budget hint for chunk-tracking branch scheduling.
- `PARALLEL_TRACKING_CHUNK_MAX_WORKERS=<int>`
  - Maximum chunk extraction workers inside the chunk-tracking branch.
- `PARALLEL_TRACKING_MAX_ENTITIES=<int>`
  - Hard cap for `summary_v2` entity count in `tracks.video_summary.json` (default: `20`).

Safe defaults are CPU-aware and conservative to avoid starvation on small hosts.

## Strategy Extension Points

- `PARALLEL_TRACKING_BACKEND_STRATEGY=default`
- `PARALLEL_TRACKING_STITCH_STRATEGY=default`
- `PARALLEL_TRACKING_ZONE_STRATEGY=grid3x3`

Unknown strategy identifiers fail fast with an explicit configuration error.

## Output Modes

- `PARALLEL_TRACKING_OUTPUT_MODE=summary_v2` (default)
  - Emits video-level simplified payload:
    - `zone_definition` (fixed 3x3 taxonomy + per-video coordinates)
    - `entities` (unique IDs, appearance ranges, zone occupancy/transitions)
  - Writes `tracks.video_summary.json`.
- `PARALLEL_TRACKING_OUTPUT_MODE=dual`
  - API payload uses `summary_v2`.
  - Also writes rollback compare artifact `tracks.compact.json`.
- `PARALLEL_TRACKING_OUTPUT_MODE=legacy`
  - API payload returns legacy compact `tracks` + `scenes`.
  - Writes `tracks.compact.json`.

## Ground-Truth Persistence Backend

- `PARALLEL_TRACKING_GROUND_TRUTH_BACKEND=sqlite` (default)
- `PARALLEL_TRACKING_GROUND_TRUTH_BACKEND=parquet` (requires `pyarrow`)

Both backends preserve canonical stitched rows:
`(t_ms, global_id, class_id, conf, x1, y1, x2, y2)`.

## Partial Success Contract

`GET /results/{job_id}` keeps `frames` semantics unchanged and adds branch-level
metadata so frame-analysis success can still be returned when chunk tracking is
disabled or fails.

## Rollback

If downstream consumers are not ready for `summary_v2`:

```bash
PARALLEL_TRACKING_OUTPUT_MODE=legacy
```

This restores prior compact payloads without disabling chunk tracking.
