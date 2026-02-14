"""Corpus construction stage for retrieval products."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from app.schemas import CorpusArtifacts, JobCorpusResult, RetrievalChunkRecord, RetrievalCorpusBundle
from app.storage import build_corpus_key

if TYPE_CHECKING:
    from app.storage import MediaStore


def _deterministic_id(kind: str, *parts: Any) -> str:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{kind}_{digest}"


def _build_retrieval_bundle(
    *,
    job_id: str,
    frame_results: list[dict[str, Any]],
    scene_outputs: dict[str, Any],
) -> RetrievalCorpusBundle:
    chunks: list[RetrievalChunkRecord] = []

    for scene in scene_outputs.get("scene_narratives", []):
        scene_id = int(scene.get("scene_id", 0))
        start_sec = float(scene.get("start_sec", 0.0))
        end_sec = float(scene.get("end_sec", 0.0))
        corpus = scene.get("corpus") or {}
        scene_chunks = corpus.get("retrieval_chunks", [])
        if scene_chunks:
            for raw in scene_chunks:
                text = str(raw.get("text", "")).strip()
                if not text:
                    continue
                chunk_id = str(raw.get("chunk_id", "")) or _deterministic_id(
                    "chunk", job_id, scene_id, text
                )
                chunks.append(
                    RetrievalChunkRecord(
                        chunk_id=chunk_id,
                        text=text,
                        metadata={
                            "job_id": job_id,
                            "scene_id": scene_id,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "artifact_keys": list(raw.get("artifact_keys", [])),
                            "source_entity_ids": list(raw.get("source_entity_ids", [])),
                        },
                    )
                )
            continue

        fallback_text = str(scene.get("narrative_paragraph", "")).strip()
        if not fallback_text:
            continue
        chunks.append(
            RetrievalChunkRecord(
                chunk_id=_deterministic_id("chunk", job_id, scene_id, fallback_text),
                text=fallback_text,
                metadata={
                    "job_id": job_id,
                    "scene_id": scene_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "artifact_keys": [scene.get("artifacts", {}).get("narrative", "")],
                },
            )
        )

    unique_chunks: dict[str, RetrievalChunkRecord] = {item.chunk_id: item for item in chunks}
    if not unique_chunks:
        for frame in frame_results:
            frame_id = int(frame.get("frame_id", 0))
            timestamp = str(frame.get("timestamp", ""))
            analysis = frame.get("analysis", {})
            labels = [str(item.get("label", "unknown")) for item in analysis.get("object_detection", [])]
            faces = [str(item.get("identity_id", "face")) for item in analysis.get("face_recognition", [])]
            descriptor = ", ".join(labels[:6]) or "no_objects"
            face_descriptor = ", ".join(faces[:4]) or "no_faces"
            text = (
                f"Frame {frame_id} at {timestamp} includes objects [{descriptor}] "
                f"and faces [{face_descriptor}]."
            )
            chunk_id = _deterministic_id("chunk", job_id, frame_id, text)
            artifacts = frame.get("analysis_artifacts", {})
            unique_chunks[chunk_id] = RetrievalChunkRecord(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    "job_id": job_id,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "artifact_keys": [str(artifacts.get("json", ""))],
                },
            )
    return RetrievalCorpusBundle(job_id=job_id, chunks=list(unique_chunks.values()))


def _persist_bundle(
    *,
    media_store: "MediaStore | None",
    job_id: str,
    artifact_kind: str,
    payload: dict[str, Any],
    filename: str = "bundle.json",
) -> str:
    if media_store is not None and hasattr(media_store, "upload_corpus_artifact"):
        return media_store.upload_corpus_artifact(
            job_id=job_id,
            artifact_kind=artifact_kind,  # type: ignore[arg-type]
            payload=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            filename=filename,
        )
    return build_corpus_key(job_id, artifact_kind, filename=filename)  # type: ignore[arg-type]


def build(
    *,
    job_id: str,
    scenes: list[tuple[float, float]],
    frame_results: list[dict[str, Any]],
    scene_outputs: dict[str, Any],
    settings: Any,
    media_store: "MediaStore | None" = None,
    embedding_client: Any | None = None,
) -> dict[str, Any]:
    """Build retrieval corpus products."""
    del scenes, settings, embedding_client
    retrieval_bundle = _build_retrieval_bundle(
        job_id=job_id,
        frame_results=frame_results,
        scene_outputs=scene_outputs,
    )

    retrieval_payload = retrieval_bundle.model_dump(mode="json")
    retrieval_key = _persist_bundle(
        media_store=media_store,
        job_id=job_id,
        artifact_kind="retrieval",
        payload=retrieval_payload,
    )

    artifacts = CorpusArtifacts(
        retrieval_bundle=retrieval_key,
    )
    result = JobCorpusResult(
        retrieval=retrieval_bundle,
        artifacts=artifacts,
    )
    return result.model_dump(mode="json")
