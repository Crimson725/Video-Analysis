"""Neo4j property graph writer for SceneGraphDelta upsert."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WriteMetrics:
    """Metrics from a single scene upsert operation."""

    nodes_created: int = 0
    nodes_updated: int = 0
    relationships_created: int = 0
    relationships_updated: int = 0
    errors: list[str] = field(default_factory=list)


class Neo4jWriter:
    """Manages Neo4j driver and upserts SceneGraphDelta data as a property graph."""

    def __init__(
        self, uri: str, user: str, password: str, database: str = "neo4j"
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    def ensure_constraints(self) -> None:
        """Drop old constraints and create new uniqueness constraints keyed on *_uid fields."""
        # Old constraints to drop (idempotent)
        old_constraints = [
            "video_id_unique",
            "scene_id_unique",
            "frame_id_unique",
            "person_id_unique",
            "object_id_unique",
            "event_id_unique",
        ]

        # New constraints keyed on *_uid fields
        new_constraints = [
            ("video_id_unique", "Video", "video_id"),
            ("scene_uid_unique", "Scene", "scene_uid"),
            ("frame_uid_unique", "Frame", "frame_uid"),
            ("person_uid_unique", "Person", "entity_uid"),
            ("object_uid_unique", "Object", "entity_uid"),
            ("event_uid_unique", "Event", "event_uid"),
        ]

        with self._driver.session(database=self._database) as session:
            # Drop old constraints
            for name in old_constraints:
                session.run(f"DROP CONSTRAINT {name} IF EXISTS")

            # Create new constraints
            for name, label, prop in new_constraints:
                cypher = (
                    f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
                session.run(cypher)

    def upsert_scene(
        self,
        *,
        video_id: str,
        scene_uid: str,
        scene_index: int,
        job_id: str,
        source_key: str,
        t0: float,
        t1: float,
        selected_frame_ids: list[int],
        frame_timestamps: dict[int, float],
        frame_uids: dict[int, str],
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        events: list[dict[str, Any]],
        possibly_same_as: list[dict[str, Any]] | None = None,
    ) -> WriteMetrics:
        """Upsert all nodes and relationships for one scene in a single transaction.

        Entity dicts must include ``entity_uid`` and ``entity_local_id``.
        Event dicts must include ``event_uid`` and local ``event_id``.
        """
        metrics = WriteMetrics()

        try:
            with self._driver.session(database=self._database) as session:
                with session.begin_transaction() as tx:
                    # 1. Video + Scene nodes
                    self._upsert_video_scene(
                        tx,
                        video_id=video_id,
                        scene_uid=scene_uid,
                        scene_index=scene_index,
                        job_id=job_id,
                        source_key=source_key,
                        t0=t0,
                        t1=t1,
                        metrics=metrics,
                    )

                    # 2. Frame nodes
                    self._upsert_frames(
                        tx,
                        scene_uid=scene_uid,
                        frame_ids=selected_frame_ids,
                        frame_timestamps=frame_timestamps,
                        frame_uids=frame_uids,
                        job_id=job_id,
                        source_key=source_key,
                        metrics=metrics,
                    )

                    # 3. Person and Object nodes + APPEARS_IN
                    self._upsert_entities(
                        tx,
                        scene_uid=scene_uid,
                        entities=entities,
                        job_id=job_id,
                        t0=t0,
                        t1=t1,
                        metrics=metrics,
                    )

                    # 4. Event nodes + HAS_EVENT + INVOLVES
                    self._upsert_events(
                        tx,
                        scene_uid=scene_uid,
                        events=events,
                        job_id=job_id,
                        source_key=source_key,
                        metrics=metrics,
                    )

                    # 5. REL relationships
                    self._upsert_relations(
                        tx,
                        scene_uid=scene_uid,
                        relations=relations,
                        metrics=metrics,
                    )

                    # 6. POSSIBLY_SAME_AS edges
                    if possibly_same_as:
                        self._upsert_possibly_same_as(
                            tx,
                            possibly_same_as=possibly_same_as,
                            scene_uid=scene_uid,
                            metrics=metrics,
                        )

                    tx.commit()

        except Exception as exc:
            metrics.errors.append(str(exc))
            logger.error("Neo4j upsert failed for scene %s: %s", scene_uid, exc)

        return metrics

    # ------------------------------------------------------------------
    # Video + Scene
    # ------------------------------------------------------------------

    def _upsert_video_scene(
        self,
        tx: Any,
        *,
        video_id: str,
        scene_uid: str,
        scene_index: int,
        job_id: str,
        source_key: str,
        t0: float,
        t1: float,
        metrics: WriteMetrics,
    ) -> None:
        result = tx.run(
            """
            MERGE (v:Video {video_id: $video_id})
            MERGE (s:Scene {scene_uid: $scene_uid})
            SET s.scene_index = $scene_index,
                s.job_id = $job_id, s.source_key = $source_key,
                s.t0 = $t0, s.t1 = $t1
            MERGE (v)-[:HAS_SCENE]->(s)
            RETURN
                v.video_id AS vid,
                s.scene_uid AS suid
            """,
            video_id=video_id,
            scene_uid=scene_uid,
            scene_index=scene_index,
            job_id=job_id,
            source_key=source_key,
            t0=t0,
            t1=t1,
        )
        summary = result.consume()
        metrics.nodes_created += summary.counters.nodes_created
        metrics.nodes_updated += max(0, 2 - summary.counters.nodes_created)
        metrics.relationships_created += summary.counters.relationships_created

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def _upsert_frames(
        self,
        tx: Any,
        *,
        scene_uid: str,
        frame_ids: list[int],
        frame_timestamps: dict[int, float],
        frame_uids: dict[int, str],
        job_id: str,
        source_key: str,
        metrics: WriteMetrics,
    ) -> None:
        if not frame_ids:
            return
        params = [
            {
                "frame_uid": frame_uids[fid],
                "frame_id": fid,
                "scene_uid": scene_uid,
                "timestamp": frame_timestamps.get(fid, 0.0),
                "job_id": job_id,
                "source_key": source_key,
            }
            for fid in frame_ids
            if fid in frame_uids
        ]
        result = tx.run(
            """
            UNWIND $frames AS f
            MATCH (s:Scene {scene_uid: f.scene_uid})
            MERGE (fr:Frame {frame_uid: f.frame_uid})
            SET fr.frame_id = f.frame_id, fr.scene_uid = f.scene_uid,
                fr.timestamp = f.timestamp,
                fr.job_id = f.job_id, fr.source_key = f.source_key
            MERGE (s)-[:HAS_FRAME]->(fr)
            """,
            frames=params,
        )
        summary = result.consume()
        metrics.nodes_created += summary.counters.nodes_created
        metrics.relationships_created += summary.counters.relationships_created

    # ------------------------------------------------------------------
    # Entities (Person / Object)
    # ------------------------------------------------------------------

    def _upsert_entities(
        self,
        tx: Any,
        *,
        scene_uid: str,
        entities: list[dict[str, Any]],
        job_id: str,
        t0: float,
        t1: float,
        metrics: WriteMetrics,
    ) -> None:
        persons = [e for e in entities if e.get("type") == "Person"]
        objects = [e for e in entities if e.get("type") != "Person"]

        if persons:
            for p in persons:
                attrs = p.get("attributes", {})
                param: dict[str, Any] = {
                    "entity_uid": p["entity_uid"],
                    "entity_local_id": p["entity_local_id"],
                    "job_id": job_id,
                    "scene_uid": scene_uid,
                    "t0": t0,
                    "t1": t1,
                    "confidence": p.get("confidence", 0.0),
                }
                for attr_key, attr_val in attrs.items():
                    if isinstance(attr_val, (str, int, float, bool)):
                        param[f"attr_{attr_key}"] = attr_val
                    else:
                        param[f"attr_{attr_key}"] = json.dumps(attr_val, default=str)

                attr_set_clauses = ", ".join(
                    f"n.{k.removeprefix('attr_')} = ${k}"
                    for k in param
                    if k.startswith("attr_")
                )
                set_clause = f"SET {attr_set_clauses}" if attr_set_clauses else ""
                result = tx.run(
                    f"""
                    MERGE (n:Person {{entity_uid: $entity_uid}})
                    SET n.entity_local_id = $entity_local_id, n.job_id = $job_id
                    {set_clause}
                    WITH n
                    MATCH (s:Scene {{scene_uid: $scene_uid}})
                    MERGE (n)-[r:APPEARS_IN]->(s)
                    SET r.scene_uid = $scene_uid, r.t0 = $t0, r.t1 = $t1,
                        r.confidence = $confidence
                    """,
                    **param,
                )
                summary = result.consume()
                metrics.nodes_created += summary.counters.nodes_created
                metrics.nodes_updated += max(0, 1 - summary.counters.nodes_created)
                metrics.relationships_created += summary.counters.relationships_created

        if objects:
            for o in objects:
                attrs = o.get("attributes", {})
                param = {
                    "entity_uid": o["entity_uid"],
                    "entity_local_id": o["entity_local_id"],
                    "label": attrs.get("label", o.get("label", "unknown")),
                    "job_id": job_id,
                    "scene_uid": scene_uid,
                    "t0": t0,
                    "t1": t1,
                    "confidence": o.get("confidence", 0.0),
                }
                for attr_key, attr_val in attrs.items():
                    if attr_key == "label":
                        continue
                    if isinstance(attr_val, (str, int, float, bool)):
                        param[f"attr_{attr_key}"] = attr_val
                    else:
                        param[f"attr_{attr_key}"] = json.dumps(attr_val, default=str)

                attr_set_clauses = ", ".join(
                    f"n.{k.removeprefix('attr_')} = ${k}"
                    for k in param
                    if k.startswith("attr_")
                )
                extra_set = f", {attr_set_clauses}" if attr_set_clauses else ""
                result = tx.run(
                    f"""
                    MERGE (n:Object {{entity_uid: $entity_uid}})
                    SET n.entity_local_id = $entity_local_id,
                        n.label = $label, n.job_id = $job_id{extra_set}
                    WITH n
                    MATCH (s:Scene {{scene_uid: $scene_uid}})
                    MERGE (n)-[r:APPEARS_IN]->(s)
                    SET r.scene_uid = $scene_uid, r.t0 = $t0, r.t1 = $t1,
                        r.confidence = $confidence
                    """,
                    **param,
                )
                summary = result.consume()
                metrics.nodes_created += summary.counters.nodes_created
                metrics.nodes_updated += max(0, 1 - summary.counters.nodes_created)
                metrics.relationships_created += summary.counters.relationships_created

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _upsert_events(
        self,
        tx: Any,
        *,
        scene_uid: str,
        events: list[dict[str, Any]],
        job_id: str,
        source_key: str,
        metrics: WriteMetrics,
    ) -> None:
        if not events:
            return
        event_params = []
        for ev in events:
            event_params.append(
                {
                    "event_uid": ev["event_uid"],
                    "event_id": ev["event_id"],
                    "event_type": ev["event_type"],
                    "t0": ev.get("time_span_s", [0, 0])[0]
                    if isinstance(ev.get("time_span_s"), (list, tuple))
                    else 0,
                    "t1": ev.get("time_span_s", [0, 0])[1]
                    if isinstance(ev.get("time_span_s"), (list, tuple))
                    else 0,
                    "summary": ev.get("summary", ""),
                    "confidence": ev.get("confidence", 0.0),
                    "scene_uid": scene_uid,
                    "job_id": job_id,
                    "source_key": source_key,
                }
            )
        result = tx.run(
            """
            UNWIND $events AS e
            MERGE (ev:Event {event_uid: e.event_uid})
            SET ev.event_id = e.event_id,
                ev.event_type = e.event_type, ev.t0 = e.t0, ev.t1 = e.t1,
                ev.summary = e.summary, ev.confidence = e.confidence,
                ev.job_id = e.job_id, ev.source_key = e.source_key
            WITH ev, e
            MATCH (s:Scene {scene_uid: e.scene_uid})
            MERGE (s)-[:HAS_EVENT]->(ev)
            """,
            events=event_params,
        )
        summary = result.consume()
        metrics.nodes_created += summary.counters.nodes_created
        metrics.relationships_created += summary.counters.relationships_created

        # INVOLVES relationships — look up entities by entity_uid
        for ev in events:
            for participant in ev.get("participants", []):
                entity_uid = participant.get("entity_uid", "")
                role = participant.get("role", "")
                if not entity_uid:
                    continue
                result = tx.run(
                    """
                    MATCH (ev:Event {event_uid: $event_uid})
                    OPTIONAL MATCH (p:Person {entity_uid: $entity_uid})
                    OPTIONAL MATCH (o:Object {entity_uid: $entity_uid})
                    WITH ev, coalesce(p, o) AS entity
                    WHERE entity IS NOT NULL
                    MERGE (ev)-[r:INVOLVES]->(entity)
                    SET r.role = $role, r.scene_uid = $scene_uid
                    """,
                    event_uid=ev["event_uid"],
                    entity_uid=entity_uid,
                    role=role,
                    scene_uid=scene_uid,
                )
                inv_summary = result.consume()
                metrics.relationships_created += (
                    inv_summary.counters.relationships_created
                )

    # ------------------------------------------------------------------
    # REL relationships
    # ------------------------------------------------------------------

    def _upsert_relations(
        self,
        tx: Any,
        *,
        scene_uid: str,
        relations: list[dict[str, Any]],
        metrics: WriteMetrics,
    ) -> None:
        if not relations:
            return
        for rel in relations:
            subject_uid = rel.get("subject_uid", "")
            object_uid = rel.get("object_uid", "")
            predicate = rel.get("predicate", "")
            t0 = (
                rel.get("time_span_s", [0, 0])[0]
                if isinstance(rel.get("time_span_s"), (list, tuple))
                else 0
            )
            t1 = (
                rel.get("time_span_s", [0, 0])[1]
                if isinstance(rel.get("time_span_s"), (list, tuple))
                else 0
            )
            confidence = rel.get("confidence", 0.0)
            evidence = json.dumps(rel.get("evidence", {}), default=str)

            result = tx.run(
                """
                OPTIONAL MATCH (s:Person {entity_uid: $subject_uid})
                OPTIONAL MATCH (s2:Object {entity_uid: $subject_uid})
                WITH coalesce(s, s2) AS subject
                WHERE subject IS NOT NULL
                OPTIONAL MATCH (o:Person {entity_uid: $object_uid})
                OPTIONAL MATCH (o2:Object {entity_uid: $object_uid})
                WITH subject, coalesce(o, o2) AS object
                WHERE object IS NOT NULL
                MERGE (subject)-[r:REL {scene_uid: $scene_uid, predicate: $predicate,
                                         t0: $t0, t1: $t1}]->(object)
                SET r.confidence = $confidence, r.evidence = $evidence
                """,
                subject_uid=subject_uid,
                object_uid=object_uid,
                predicate=predicate,
                scene_uid=scene_uid,
                t0=t0,
                t1=t1,
                confidence=confidence,
                evidence=evidence,
            )
            rel_summary = result.consume()
            metrics.relationships_created += rel_summary.counters.relationships_created

    # ------------------------------------------------------------------
    # POSSIBLY_SAME_AS
    # ------------------------------------------------------------------

    def _upsert_possibly_same_as(
        self,
        tx: Any,
        *,
        possibly_same_as: list[dict[str, Any]],
        scene_uid: str,
        metrics: WriteMetrics,
    ) -> None:
        for link in possibly_same_as:
            entity_a_uid = link.get("entity_a_uid", "")
            entity_b_uid = link.get("entity_b_uid", "")
            confidence = link.get("confidence", 0.0)
            evidence = json.dumps(link.get("evidence", {}), default=str)

            result = tx.run(
                """
                OPTIONAL MATCH (a:Person {entity_uid: $entity_a_uid})
                OPTIONAL MATCH (a2:Object {entity_uid: $entity_a_uid})
                WITH coalesce(a, a2) AS nodeA
                WHERE nodeA IS NOT NULL
                OPTIONAL MATCH (b:Person {entity_uid: $entity_b_uid})
                OPTIONAL MATCH (b2:Object {entity_uid: $entity_b_uid})
                WITH nodeA, coalesce(b, b2) AS nodeB
                WHERE nodeB IS NOT NULL
                MERGE (nodeA)-[r:POSSIBLY_SAME_AS]->(nodeB)
                SET r.confidence = $confidence, r.scene_uid = $scene_uid,
                    r.evidence = $evidence
                """,
                entity_a_uid=entity_a_uid,
                entity_b_uid=entity_b_uid,
                confidence=confidence,
                scene_uid=scene_uid,
                evidence=evidence,
            )
            link_summary = result.consume()
            metrics.relationships_created += link_summary.counters.relationships_created
