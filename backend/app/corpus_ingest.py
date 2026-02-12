"""Graph/vector adapter interfaces and corpus ingest orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from app.schemas import JobCorpusResult


class GraphAdapter(Protocol):
    """Graph storage adapter contract."""

    def upsert_corpus(self, corpus: JobCorpusResult) -> dict[str, int]:
        """Upsert corpus graph payloads and return operation counters."""


class VectorAdapter(Protocol):
    """Vector storage adapter contract."""

    def upsert_corpus(self, corpus: JobCorpusResult) -> dict[str, int]:
        """Upsert retrieval chunks + embeddings and return operation counters."""


@dataclass
class InMemoryGraphAdapter:
    """In-memory graph adapter for tests and local dry-runs."""

    nodes: dict[str, dict[str, Any]]
    edges: dict[str, dict[str, Any]]
    claims: dict[str, dict[str, Any]]

    def __init__(self) -> None:
        self.nodes = {}
        self.edges = {}
        self.claims = {}

    def upsert_corpus(self, corpus: JobCorpusResult) -> dict[str, int]:
        for node in corpus.graph.nodes:
            self.nodes[node.node_id] = node.model_dump(mode="json")
        for edge in corpus.graph.edges:
            self.edges[edge.edge_id] = edge.model_dump(mode="json")
        for claim in corpus.graph.derived_claims:
            self.claims[claim.claim_id] = claim.model_dump(mode="json")
        return {
            "nodes": len(corpus.graph.nodes),
            "edges": len(corpus.graph.edges),
            "claims": len(corpus.graph.derived_claims),
        }


@dataclass
class InMemoryVectorAdapter:
    """In-memory vector adapter for tests and local dry-runs."""

    chunks: dict[str, dict[str, Any]]

    def __init__(self) -> None:
        self.chunks = {}

    def upsert_corpus(self, corpus: JobCorpusResult) -> dict[str, int]:
        embeddings_by_id = {item.chunk_id: item for item in corpus.embeddings.embeddings}
        for chunk in corpus.retrieval.chunks:
            embedding = embeddings_by_id.get(chunk.chunk_id)
            if embedding is None:
                continue
            self.chunks[chunk.chunk_id] = {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": embedding.vector,
                "model_id": embedding.model_id,
                "model_version": embedding.model_version,
            }
        return {
            "chunks": len(self.chunks),
        }


class Neo4jGraphAdapter:
    """Neo4j Community graph adapter for local development."""

    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str,
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except Exception as exc:  # pragma: no cover - import availability differs by env
            raise RuntimeError(
                "neo4j driver is not installed; add 'neo4j' dependency to use Neo4jGraphAdapter"
            ) from exc
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    @staticmethod
    def _merge_node(tx: Any, node: dict[str, Any]) -> None:
        tx.run(
            """
            MERGE (n:CorpusNode {node_id: $node_id})
            SET n.node_type = $node_type,
                n.label = $label,
                n.confidence = $confidence,
                n.provenance_json = $provenance_json
            """,
            node_id=node["node_id"],
            node_type=node["node_type"],
            label=node["label"],
            confidence=node["confidence"],
            provenance_json=json.dumps(node.get("provenance", {}), separators=(",", ":")),
        )

    @staticmethod
    def _merge_edge(tx: Any, edge: dict[str, Any]) -> None:
        tx.run(
            """
            MATCH (source:CorpusNode {node_id: $source_node_id})
            MATCH (target:CorpusNode {node_id: $target_node_id})
            MERGE (source)-[r:CORPUS_REL {edge_id: $edge_id}]->(target)
            SET r.predicate = $predicate,
                r.confidence = $confidence,
                r.evidence_json = $evidence_json
            """,
            edge_id=edge["edge_id"],
            source_node_id=edge["source_node_id"],
            target_node_id=edge["target_node_id"],
            predicate=edge["predicate"],
            confidence=edge["confidence"],
            evidence_json=json.dumps(edge.get("evidence", []), separators=(",", ":")),
        )

    @staticmethod
    def _merge_claim(tx: Any, claim: dict[str, Any]) -> None:
        tx.run(
            """
            MERGE (c:CorpusClaim {claim_id: $claim_id})
            SET c.text = $text,
                c.confidence = $confidence,
                c.provenance_json = $provenance_json,
                c.evidence_json = $evidence_json
            """,
            claim_id=claim["claim_id"],
            text=claim["text"],
            confidence=claim["confidence"],
            provenance_json=json.dumps(claim.get("provenance", {}), separators=(",", ":")),
            evidence_json=json.dumps(claim.get("evidence", []), separators=(",", ":")),
        )

    def upsert_corpus(self, corpus: JobCorpusResult) -> dict[str, int]:
        with self._driver.session(database=self._database) as session:
            for node in corpus.graph.nodes:
                session.execute_write(self._merge_node, node.model_dump(mode="json"))
            for edge in corpus.graph.edges:
                session.execute_write(self._merge_edge, edge.model_dump(mode="json"))
            for claim in corpus.graph.derived_claims:
                session.execute_write(self._merge_claim, claim.model_dump(mode="json"))
        return {
            "nodes": len(corpus.graph.nodes),
            "edges": len(corpus.graph.edges),
            "claims": len(corpus.graph.derived_claims),
        }


class PgVectorAdapter:
    """PostgreSQL+pgvector adapter for retrieval chunk + embedding upserts."""

    def __init__(
        self,
        *,
        dsn: str,
        embedding_dimension: int,
    ) -> None:
        self._dsn = dsn
        self._embedding_dimension = embedding_dimension

    def _connect(self) -> Any:
        try:
            import psycopg
        except Exception as exc:  # pragma: no cover - import availability differs by env
            raise RuntimeError(
                "psycopg is not installed; add 'psycopg[binary]' dependency to use PgVectorAdapter"
            ) from exc
        return psycopg.connect(self._dsn)

    def _ensure_schema(self, conn: Any) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS corpus_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self._embedding_dimension}) NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL
                )
                """
            )
        conn.commit()

    def upsert_corpus(self, corpus: JobCorpusResult) -> dict[str, int]:
        embeddings_by_id = {item.chunk_id: item for item in corpus.embeddings.embeddings}
        with self._connect() as conn:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                upserted = 0
                for chunk in corpus.retrieval.chunks:
                    embedding = embeddings_by_id.get(chunk.chunk_id)
                    if embedding is None:
                        continue
                    vector_literal = "[" + ",".join(str(value) for value in embedding.vector) + "]"
                    cur.execute(
                        """
                        INSERT INTO corpus_chunks (
                            chunk_id,
                            chunk_text,
                            metadata,
                            embedding,
                            model_id,
                            model_version
                        )
                        VALUES (
                            %(chunk_id)s,
                            %(chunk_text)s,
                            %(metadata)s::jsonb,
                            %(embedding)s::vector,
                            %(model_id)s,
                            %(model_version)s
                        )
                        ON CONFLICT (chunk_id)
                        DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            model_id = EXCLUDED.model_id,
                            model_version = EXCLUDED.model_version
                        """,
                        {
                            "chunk_id": chunk.chunk_id,
                            "chunk_text": chunk.text,
                            "metadata": json.dumps(chunk.metadata, separators=(",", ":")),
                            "embedding": vector_literal,
                            "model_id": embedding.model_id,
                            "model_version": embedding.model_version,
                        },
                    )
                    upserted += 1
                conn.commit()
        return {"chunks": upserted}


def build_graph_adapter(settings: Any) -> GraphAdapter:
    """Build configured graph adapter for ingest."""
    backend = str(getattr(settings, "graph_backend", "neo4j")).strip().lower()
    if backend == "memory":
        return InMemoryGraphAdapter()
    if backend == "neo4j":
        return Neo4jGraphAdapter(
            uri=str(getattr(settings, "neo4j_uri")),
            username=str(getattr(settings, "neo4j_username")),
            password=str(getattr(settings, "neo4j_password")),
            database=str(getattr(settings, "neo4j_database")),
        )
    raise ValueError(f"Unsupported graph backend: {backend}")


def build_vector_adapter(settings: Any) -> VectorAdapter:
    """Build configured vector adapter for ingest."""
    backend = str(getattr(settings, "vector_backend", "pgvector")).strip().lower()
    if backend == "memory":
        return InMemoryVectorAdapter()
    if backend == "pgvector":
        return PgVectorAdapter(
            dsn=str(getattr(settings, "pgvector_dsn")),
            embedding_dimension=int(getattr(settings, "embedding_dimension")),
        )
    raise ValueError(f"Unsupported vector backend: {backend}")


def ingest_corpus(
    *,
    corpus_payload: dict[str, Any],
    graph_adapter: GraphAdapter,
    vector_adapter: VectorAdapter,
) -> dict[str, Any]:
    """Idempotent upsert ingest flow for graph and vector stores."""
    corpus = JobCorpusResult.model_validate(corpus_payload)
    graph_counts = graph_adapter.upsert_corpus(corpus)
    vector_counts = vector_adapter.upsert_corpus(corpus)
    return {
        "graph": graph_counts,
        "vector": vector_counts,
    }
