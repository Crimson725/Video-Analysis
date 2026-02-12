"""Corpus construction stage for graph and retrieval products."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from app.schemas import (
    CorpusArtifacts,
    DerivedClaim,
    EmbeddingRecord,
    EmbeddingsCorpusBundle,
    EvidenceAnchor,
    GraphCorpusBundle,
    GraphEdge,
    GraphNode,
    JobCorpusResult,
    RetrievalChunkRecord,
    RetrievalCorpusBundle,
    SourceFact,
)
from app.storage import build_corpus_key

if TYPE_CHECKING:
    from app.config import Settings
    from app.storage import MediaStore


def _deterministic_id(kind: str, *parts: Any) -> str:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{kind}_{digest}"


def _anchor_from_frame(frame: dict[str, Any], bbox: list[int] | None, text_span: str | None) -> EvidenceAnchor:
    artifacts = frame.get("analysis_artifacts", {})
    artifact_key = str(artifacts.get("json", ""))
    return EvidenceAnchor(
        frame_id=int(frame.get("frame_id", 0)),
        timestamp=str(frame.get("timestamp", "")),
        artifact_key=artifact_key,
        bbox=bbox,
        text_span=text_span,
    )


def _build_graph_bundle(
    *,
    job_id: str,
    frame_results: list[dict[str, Any]],
    scene_outputs: dict[str, Any],
) -> GraphCorpusBundle:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    source_facts: list[SourceFact] = []
    derived_claims: list[DerivedClaim] = []

    node_index: dict[str, str] = {}

    for frame in frame_results:
        analysis = frame.get("analysis", {})
        for det in analysis.get("object_detection", []):
            label = str(det.get("label", "unknown"))
            track_id = str(det.get("track_id", "")) or _deterministic_id(
                "track", job_id, frame.get("frame_id"), label
            )
            node_id = node_index.setdefault(
                track_id,
                _deterministic_id("node", job_id, "object", label, track_id),
            )
            anchor = _anchor_from_frame(frame, det.get("box"), label)
            source_facts.append(
                SourceFact(
                    fact_id=_deterministic_id(
                        "fact",
                        job_id,
                        "object",
                        frame.get("frame_id"),
                        track_id,
                    ),
                    fact_type="object_detection",
                    confidence=float(det.get("confidence", 0.0)),
                    payload={
                        "label": label,
                        "track_id": track_id,
                    },
                    evidence=[anchor],
                )
            )
            nodes.append(
                GraphNode(
                    node_id=node_id,
                    node_type="object",
                    label=label,
                    confidence=float(det.get("confidence", 0.0)),
                    provenance={
                        "source": "frame_analysis",
                        "track_id": track_id,
                    },
                )
            )

        for face in analysis.get("face_recognition", []):
            identity_id = str(face.get("identity_id", "")) or _deterministic_id(
                "identity", job_id, frame.get("frame_id"), face.get("face_id")
            )
            node_id = node_index.setdefault(
                identity_id,
                _deterministic_id("node", job_id, "face", identity_id),
            )
            anchor = _anchor_from_frame(frame, face.get("coordinates"), identity_id)
            source_facts.append(
                SourceFact(
                    fact_id=_deterministic_id(
                        "fact",
                        job_id,
                        "face",
                        frame.get("frame_id"),
                        identity_id,
                    ),
                    fact_type="face_detection",
                    confidence=float(face.get("confidence", 0.0)),
                    payload={
                        "identity_id": identity_id,
                    },
                    evidence=[anchor],
                )
            )
            nodes.append(
                GraphNode(
                    node_id=node_id,
                    node_type="person",
                    label=identity_id,
                    confidence=float(face.get("confidence", 0.0)),
                    provenance={
                        "source": "frame_analysis",
                        "identity_id": identity_id,
                    },
                )
            )

    scene_narratives = scene_outputs.get("scene_narratives", [])
    for scene in scene_narratives:
        scene_id = int(scene.get("scene_id", 0))
        corpus = scene.get("corpus") or {}
        for relation in corpus.get("relations", []):
            source_entity_id = str(relation.get("source_entity_id", ""))
            target_entity_id = str(relation.get("target_entity_id", ""))
            if not source_entity_id or not target_entity_id:
                continue
            evidence = [EvidenceAnchor.model_validate(item) for item in relation.get("evidence", [])]
            if not evidence:
                evidence = [
                    EvidenceAnchor(
                        frame_id=0,
                        timestamp="00:00:00.000",
                        artifact_key=build_corpus_key(job_id, "graph", f"scene_{scene_id}.json"),
                        text_span="scene_relation",
                    )
                ]
            edges.append(
                GraphEdge(
                    edge_id=_deterministic_id(
                        "edge",
                        job_id,
                        scene_id,
                        source_entity_id,
                        relation.get("predicate", "related_to"),
                        target_entity_id,
                    ),
                    source_node_id=source_entity_id,
                    target_node_id=target_entity_id,
                    predicate=str(relation.get("predicate", "related_to")),
                    confidence=float(relation.get("confidence", 0.0)),
                    evidence=evidence,
                )
            )

        paragraph = str(scene.get("narrative_paragraph", "")).strip()
        if paragraph:
            derived_claims.append(
                DerivedClaim(
                    claim_id=_deterministic_id("claim", job_id, scene_id, paragraph),
                    text=paragraph,
                    confidence=0.7,
                    provenance={
                        "source": "scene_narrative",
                        "scene_id": scene_id,
                    },
                    evidence=[
                        EvidenceAnchor(
                            frame_id=0,
                            timestamp=f"scene_{scene_id}",
                            artifact_key=str(scene.get("artifacts", {}).get("narrative", "")),
                            text_span=paragraph[:120],
                        )
                    ],
                )
            )

    unique_nodes: dict[str, GraphNode] = {item.node_id: item for item in nodes}
    unique_edges: dict[str, GraphEdge] = {item.edge_id: item for item in edges}
    unique_source_facts: dict[str, SourceFact] = {item.fact_id: item for item in source_facts}
    unique_claims: dict[str, DerivedClaim] = {item.claim_id: item for item in derived_claims}

    if not unique_nodes and frame_results:
        frame = frame_results[0]
        frame_id = int(frame.get("frame_id", 0))
        timestamp = str(frame.get("timestamp", ""))
        artifacts = frame.get("analysis_artifacts", {})
        fallback_anchor = EvidenceAnchor(
            frame_id=frame_id,
            timestamp=timestamp,
            artifact_key=str(artifacts.get("json", "")),
            text_span="frame_observed",
        )
        fallback_node = GraphNode(
            node_id=_deterministic_id("node", job_id, "frame", frame_id),
            node_type="frame",
            label=f"frame_{frame_id}",
            confidence=1.0,
            provenance={"source": "frame_analysis_fallback"},
        )
        fallback_fact = SourceFact(
            fact_id=_deterministic_id("fact", job_id, "frame", frame_id),
            fact_type="frame_observed",
            confidence=1.0,
            payload={"frame_id": frame_id},
            evidence=[fallback_anchor],
        )
        unique_nodes[fallback_node.node_id] = fallback_node
        unique_source_facts[fallback_fact.fact_id] = fallback_fact

    return GraphCorpusBundle(
        job_id=job_id,
        nodes=list(unique_nodes.values()),
        edges=list(unique_edges.values()),
        source_facts=list(unique_source_facts.values()),
        derived_claims=list(unique_claims.values()),
    )


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


def _deterministic_embedding_values(text: str, chunk_id: str, dimension: int) -> list[float]:
    seed = hashlib.sha256(f"{chunk_id}|{text}".encode("utf-8")).digest()
    values: list[float] = []
    for index in range(dimension):
        byte = seed[index % len(seed)]
        values.append((byte / 255.0) * 2.0 - 1.0)
    return values


def _build_embeddings_bundle(
    *,
    job_id: str,
    retrieval_bundle: RetrievalCorpusBundle,
    settings: "Settings",
) -> EmbeddingsCorpusBundle:
    embeddings: list[EmbeddingRecord] = []
    for chunk in retrieval_bundle.chunks:
        embeddings.append(
            EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                vector=_deterministic_embedding_values(
                    chunk.text,
                    chunk.chunk_id,
                    settings.embedding_dimension,
                ),
                model_id=settings.embedding_model_id,
                model_version=settings.embedding_model_version,
            )
        )
    return EmbeddingsCorpusBundle(
        job_id=job_id,
        dimension=settings.embedding_dimension,
        embeddings=embeddings,
    )


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
    settings: "Settings",
    media_store: "MediaStore | None" = None,
) -> dict[str, Any]:
    """Build synchronized graph/retrieval/embeddings corpus products."""
    del scenes
    graph_bundle = _build_graph_bundle(
        job_id=job_id,
        frame_results=frame_results,
        scene_outputs=scene_outputs,
    )
    retrieval_bundle = _build_retrieval_bundle(
        job_id=job_id,
        frame_results=frame_results,
        scene_outputs=scene_outputs,
    )
    embeddings_bundle = _build_embeddings_bundle(
        job_id=job_id,
        retrieval_bundle=retrieval_bundle,
        settings=settings,
    )

    graph_payload = graph_bundle.model_dump(mode="json")
    retrieval_payload = retrieval_bundle.model_dump(mode="json")
    embeddings_payload = embeddings_bundle.model_dump(mode="json")

    graph_key = _persist_bundle(
        media_store=media_store,
        job_id=job_id,
        artifact_kind="graph",
        payload=graph_payload,
    )
    retrieval_key = _persist_bundle(
        media_store=media_store,
        job_id=job_id,
        artifact_kind="retrieval",
        payload=retrieval_payload,
    )
    embeddings_key = _persist_bundle(
        media_store=media_store,
        job_id=job_id,
        artifact_kind="embeddings",
        payload=embeddings_payload,
    )

    artifacts = CorpusArtifacts(
        graph_bundle=graph_key,
        retrieval_bundle=retrieval_key,
        embeddings_bundle=embeddings_key,
    )
    result = JobCorpusResult(
        graph=graph_bundle,
        retrieval=retrieval_bundle,
        embeddings=embeddings_bundle,
        artifacts=artifacts,
    )
    return result.model_dump(mode="json")
