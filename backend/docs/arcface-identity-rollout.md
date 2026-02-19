# ArcFace Identity Consistency Rollout (macOS)

This guide documents runtime assets, configuration, rollout tuning, and rollback
for ArcFace-based face embeddings and video-level identity consolidation.

## 1) Required assets and runtime configuration

### Model/runtime dependencies
- `insightface>=0.7.3` (currently packaged for Python `<3.13`)
- `onnxruntime>=1.17.0`

### ArcFace defaults
- Model pack: `buffalo_l`
- Provider preference order: `CoreMLExecutionProvider,CPUExecutionProvider`
- Fallback behavior: `cpu`

### Environment variables
- `ENABLE_FACE_IDENTITY_PIPELINE=true`
- `FACE_IDENTITY_ARCFACE_MODEL_NAME=buffalo_l`
- `FACE_IDENTITY_ARCFACE_PROVIDER_ORDER=CoreMLExecutionProvider,CPUExecutionProvider`
- `FACE_IDENTITY_ARCFACE_FALLBACK_BEHAVIOR=cpu`
- `FACE_IDENTITY_AMBIGUITY_MARGIN=0.03`

Optional:
- `FACE_IDENTITY_SAMPLE_FPS` (default `4`)
- `FACE_IDENTITY_MAX_SAMPLES_PER_SCENE` (default `120`)
- `FACE_IDENTITY_SCENE_SIMILARITY_THRESHOLD` (default `0.68`)
- `FACE_IDENTITY_VIDEO_SIMILARITY_THRESHOLD` (default `0.74`)

## 2) Runtime observability

The pipeline emits runtime metadata for the active face embedding backend:
- `video_face_identities.backend`
- `video_face_identities.provider_path`
- `video_face_identities.active_provider`

Per-frame metadata also includes a `face_embedder` provenance entry with provider
path details when available.

## 3) Conservative rollout guidance

Start with conservative matching to avoid false identity merges:
- Keep `FACE_IDENTITY_AMBIGUITY_MARGIN` at `0.03` or higher initially.
- Increase `FACE_IDENTITY_VIDEO_SIMILARITY_THRESHOLD` when ambiguous merges are
  high.
- Review output fields:
  - `is_identity_ambiguous` on faces and person tracks
  - `object_identity_is_ambiguous` on detections
  - `identity_confidence` on object/person summaries

Suggested rollout sequence:
1. Enable ArcFace runtime with CPU fallback in staging.
2. Validate provider path (`CoreMLExecutionProvider` preferred on Apple Silicon).
3. Compare identity stability across representative videos (occlusions, re-entry,
   chunk boundaries).
4. Promote to production with threshold tuning based on ambiguity/merge quality.

## 4) Rollback guidance

To disable new identity consolidation and ArcFace runtime paths:
- Set `ENABLE_FACE_IDENTITY_PIPELINE=false`.

To keep face identity enabled but force non-CoreML fallback behavior:
- Set `FACE_IDENTITY_ARCFACE_PROVIDER_ORDER=CPUExecutionProvider`.

To reduce identity linking aggressiveness without fully disabling:
- Raise `FACE_IDENTITY_VIDEO_SIMILARITY_THRESHOLD`.
- Raise `FACE_IDENTITY_AMBIGUITY_MARGIN`.

These controls preserve baseline frame analysis while reducing or removing
ArcFace-driven identity fusion behavior.
