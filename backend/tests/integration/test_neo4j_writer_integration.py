"""Integration tests for Neo4jWriter (requires local Neo4j via `make up`)."""

import pytest

from app.neo4j_writer import Neo4jWriter, WriteMetrics

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "local-dev-password"
NEO4J_DATABASE = "neo4j"


@pytest.fixture()
def writer():
    """Create a Neo4jWriter connected to local Neo4j."""
    w = Neo4jWriter(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
    yield w
    # Cleanup: delete all test data
    try:
        with w._driver.session(database=NEO4J_DATABASE) as session:
            session.run(
                "MATCH (n) WHERE n.scene_uid STARTS WITH 'test_' "
                "OR n.video_id STARTS WITH 'test_' "
                "OR n.entity_uid STARTS WITH 'test_' "
                "OR n.event_uid STARTS WITH 'test_' "
                "OR n.frame_uid STARTS WITH 'test_' "
                "DETACH DELETE n"
            )
    except Exception:
        pass
    w.close()


@pytest.mark.integration
class TestNeo4jConstraints:
    def test_ensure_constraints_creates_constraints(self, writer: Neo4jWriter):
        writer.ensure_constraints()
        with writer._driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = [r.data() for r in result]
        # Should have 6 constraints keyed on *_uid fields
        assert len(constraints) >= 6
        constraint_names = {c.get("name", "") for c in constraints}
        assert "scene_uid_unique" in constraint_names
        assert "frame_uid_unique" in constraint_names
        assert "person_uid_unique" in constraint_names
        assert "object_uid_unique" in constraint_names
        assert "event_uid_unique" in constraint_names
        assert "video_id_unique" in constraint_names


@pytest.mark.integration
class TestNeo4jUpsertScene:
    def test_basic_upsert(self, writer: Neo4jWriter):
        writer.ensure_constraints()
        metrics = writer.upsert_scene(
            video_id="test_vid_001",
            scene_uid="test_vid_001:0",
            scene_index=0,
            job_id="test_vid_001",
            source_key="input/test.mp4",
            t0=0.0,
            t1=5.0,
            selected_frame_ids=[0, 3, 5],
            frame_timestamps={0: 0.0, 3: 1.5, 5: 2.5},
            frame_uids={0: "test_vid_001:0:0", 3: "test_vid_001:0:3", 5: "test_vid_001:0:5"},
            entities=[
                {
                    "entity_uid": "test_vid_001:person_7",
                    "entity_local_id": "person_7",
                    "type": "Person",
                    "attributes": {"clothing": ["dark jacket"]},
                    "confidence": 0.9,
                },
                {
                    "entity_uid": "test_vid_001:obj_phone",
                    "entity_local_id": "obj_phone",
                    "type": "Object",
                    "attributes": {"label": "phone"},
                    "confidence": 0.85,
                },
            ],
            relations=[
                {
                    "subject_uid": "test_vid_001:person_7",
                    "object_uid": "test_vid_001:obj_phone",
                    "predicate": "holding",
                    "time_span_s": [1.0, 2.4],
                    "confidence": 0.9,
                    "evidence": {"supporting_frames": [0, 3]},
                },
            ],
            events=[
                {
                    "event_uid": "test_vid_001:0:ev_001",
                    "event_id": "ev_001",
                    "event_type": "handoff_object",
                    "participants": [
                        {"entity_uid": "test_vid_001:person_7", "entity_id": "person_7", "role": "giver"},
                    ],
                    "time_span_s": [1.8, 2.2],
                    "summary": "Object handoff",
                    "confidence": 0.8,
                },
            ],
        )
        assert isinstance(metrics, WriteMetrics)
        assert len(metrics.errors) == 0
        assert metrics.nodes_created > 0 or metrics.nodes_updated > 0

    def test_idempotent_upsert(self, writer: Neo4jWriter):
        """Upserting the same scene twice should not duplicate nodes."""
        writer.ensure_constraints()
        params = dict(
            video_id="test_vid_idem",
            scene_uid="test_vid_idem:0",
            scene_index=0,
            job_id="test_vid_idem",
            source_key="input/test.mp4",
            t0=0.0,
            t1=3.0,
            selected_frame_ids=[0, 1],
            frame_timestamps={0: 0.0, 1: 1.0},
            frame_uids={0: "test_vid_idem:0:0", 1: "test_vid_idem:0:1"},
            entities=[
                {
                    "entity_uid": "test_vid_idem:person_idem",
                    "entity_local_id": "person_idem",
                    "type": "Person",
                    "attributes": {},
                    "confidence": 0.9,
                },
            ],
            relations=[],
            events=[],
        )
        writer.upsert_scene(**params)
        metrics2 = writer.upsert_scene(**params)
        # Second upsert should not create new nodes (all merges)
        assert len(metrics2.errors) == 0

        # Verify only one scene node exists
        with writer._driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                "MATCH (s:Scene {scene_uid: $suid}) RETURN count(s) AS cnt",
                suid="test_vid_idem:0",
            )
            count = result.single()["cnt"]
        assert count == 1

    def test_relationship_upsert(self, writer: Neo4jWriter):
        writer.ensure_constraints()
        metrics = writer.upsert_scene(
            video_id="test_vid_rel",
            scene_uid="test_vid_rel:0",
            scene_index=0,
            job_id="test_vid_rel",
            source_key="input/test.mp4",
            t0=0.0,
            t1=5.0,
            selected_frame_ids=[0],
            frame_timestamps={0: 0.0},
            frame_uids={0: "test_vid_rel:0:0"},
            entities=[
                {"entity_uid": "test_vid_rel:p_a", "entity_local_id": "p_a", "type": "Person", "attributes": {}, "confidence": 0.9},
                {"entity_uid": "test_vid_rel:p_b", "entity_local_id": "p_b", "type": "Person", "attributes": {}, "confidence": 0.8},
            ],
            relations=[
                {
                    "subject_uid": "test_vid_rel:p_a",
                    "object_uid": "test_vid_rel:p_b",
                    "predicate": "near",
                    "time_span_s": [0.0, 1.0],
                    "confidence": 0.7,
                    "evidence": {"supporting_frames": [0]},
                },
                {
                    "subject_uid": "test_vid_rel:p_a",
                    "object_uid": "test_vid_rel:p_b",
                    "predicate": "speaking_to",
                    "time_span_s": [1.0, 2.0],
                    "confidence": 0.6,
                    "evidence": {"supporting_frames": [0]},
                },
            ],
            events=[],
        )
        assert len(metrics.errors) == 0

    def test_metrics_returned(self, writer: Neo4jWriter):
        writer.ensure_constraints()
        metrics = writer.upsert_scene(
            video_id="test_vid_metrics",
            scene_uid="test_vid_metrics:0",
            scene_index=0,
            job_id="test_vid_metrics",
            source_key="input/test.mp4",
            t0=0.0,
            t1=2.0,
            selected_frame_ids=[0],
            frame_timestamps={0: 0.0},
            frame_uids={0: "test_vid_metrics:0:0"},
            entities=[
                {"entity_uid": "test_vid_metrics:p_met", "entity_local_id": "p_met", "type": "Person", "attributes": {}, "confidence": 0.9},
            ],
            relations=[],
            events=[],
        )
        assert isinstance(metrics.nodes_created, int)
        assert isinstance(metrics.nodes_updated, int)
        assert isinstance(metrics.relationships_created, int)
        assert isinstance(metrics.errors, list)


@pytest.mark.integration
class TestMultiJobIsolation:
    """Two different job_ids with scene_index=0 must create distinct nodes."""

    def test_distinct_jobs_create_distinct_nodes(self, writer: Neo4jWriter):
        writer.ensure_constraints()

        base_params = dict(
            source_key="input/test.mp4",
            t0=0.0,
            t1=5.0,
            selected_frame_ids=[0],
            frame_timestamps={0: 0.0},
            relations=[],
            events=[],
        )

        # Job A
        writer.upsert_scene(
            video_id="test_job_A",
            scene_uid="test_job_A:0",
            scene_index=0,
            job_id="test_job_A",
            frame_uids={0: "test_job_A:0:0"},
            entities=[
                {"entity_uid": "test_job_A:person_1", "entity_local_id": "person_1", "type": "Person", "attributes": {}, "confidence": 0.9},
            ],
            **base_params,
        )

        # Job B — same scene_index but different job_id
        writer.upsert_scene(
            video_id="test_job_B",
            scene_uid="test_job_B:0",
            scene_index=0,
            job_id="test_job_B",
            frame_uids={0: "test_job_B:0:0"},
            entities=[
                {"entity_uid": "test_job_B:person_1", "entity_local_id": "person_1", "type": "Person", "attributes": {}, "confidence": 0.9},
            ],
            **base_params,
        )

        with writer._driver.session(database=NEO4J_DATABASE) as session:
            # Two distinct Scene nodes
            result = session.run(
                "MATCH (s:Scene) WHERE s.scene_uid IN ['test_job_A:0', 'test_job_B:0'] "
                "RETURN count(s) AS cnt"
            )
            assert result.single()["cnt"] == 2

            # Two distinct Frame nodes
            result = session.run(
                "MATCH (f:Frame) WHERE f.frame_uid IN ['test_job_A:0:0', 'test_job_B:0:0'] "
                "RETURN count(f) AS cnt"
            )
            assert result.single()["cnt"] == 2

            # Two distinct Person nodes
            result = session.run(
                "MATCH (p:Person) WHERE p.entity_uid IN ['test_job_A:person_1', 'test_job_B:person_1'] "
                "RETURN count(p) AS cnt"
            )
            assert result.single()["cnt"] == 2


@pytest.mark.integration
class TestIdempotency:
    """Ingesting the same job_id + scene_index twice must not change node count."""

    def test_idempotent_reingest(self, writer: Neo4jWriter):
        writer.ensure_constraints()
        params = dict(
            video_id="test_idem_job",
            scene_uid="test_idem_job:0",
            scene_index=0,
            job_id="test_idem_job",
            source_key="input/test.mp4",
            t0=0.0,
            t1=5.0,
            selected_frame_ids=[0, 1],
            frame_timestamps={0: 0.0, 1: 1.0},
            frame_uids={0: "test_idem_job:0:0", 1: "test_idem_job:0:1"},
            entities=[
                {"entity_uid": "test_idem_job:p1", "entity_local_id": "p1", "type": "Person", "attributes": {}, "confidence": 0.9},
                {"entity_uid": "test_idem_job:obj1", "entity_local_id": "obj1", "type": "Object", "attributes": {"label": "phone"}, "confidence": 0.8},
            ],
            relations=[],
            events=[
                {
                    "event_uid": "test_idem_job:0:ev1",
                    "event_id": "ev1",
                    "event_type": "interaction",
                    "participants": [],
                    "time_span_s": [0.0, 1.0],
                    "summary": "test",
                    "confidence": 0.7,
                },
            ],
        )

        # First ingest
        writer.upsert_scene(**params)

        with writer._driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                "MATCH (n) WHERE n.job_id = 'test_idem_job' RETURN count(n) AS cnt"
            )
            count_after_first = result.single()["cnt"]

        # Second ingest (identical)
        metrics2 = writer.upsert_scene(**params)
        assert len(metrics2.errors) == 0

        with writer._driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                "MATCH (n) WHERE n.job_id = 'test_idem_job' RETURN count(n) AS cnt"
            )
            count_after_second = result.single()["cnt"]

        assert count_after_second == count_after_first
