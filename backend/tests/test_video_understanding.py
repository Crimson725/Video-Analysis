"""Unit tests for app.video_understanding."""

import json
from types import SimpleNamespace

import pytest

from app.scene_packet_builder import ScenePacketValidationError, build_scene_packets
from app.scene_worker_runtime import PromptPolicy, build_scene_multimodal_messages, build_scene_prompt
from app.storage import build_scene_key, build_summary_key
from app.video_understanding import GeminiSceneLLMClient, run_scene_understanding_pipeline


def _settings(**overrides):
    defaults = {
        "google_api_key": "",
        "scene_model_id": "gemini-3-flash-preview",
        "synopsis_model_id": "gemini-3-flash-preview",
        "scene_llm_retry_count": 1,
        "scene_packet_max_entities": 8,
        "scene_packet_max_events": 8,
        "scene_packet_max_keyframes": 3,
        "scene_packet_disambiguation_label_threshold": 2,
        "langsmith_tracing_enabled": False,
        "langsmith_project": "",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeMediaStore:
    """Minimal media store stub for scene pipeline tests."""

    def __init__(self):
        self.scene_uploads: list[tuple[str, int, bytes]] = []
        self.summary_uploads: list[tuple[str, bytes]] = []

    def upload_scene_artifact(self, job_id: str, artifact_kind: str, scene_id: int, payload: bytes) -> str:
        key = build_scene_key(job_id, artifact_kind, scene_id)  # type: ignore[arg-type]
        self.scene_uploads.append((artifact_kind, scene_id, payload))
        return key

    def upload_summary_artifact(self, job_id: str, artifact_kind: str, payload: bytes) -> str:
        key = build_summary_key(job_id, artifact_kind)  # type: ignore[arg-type]
        self.summary_uploads.append((artifact_kind, payload))
        return key


class RecordingLLM:
    """LLM stub with call/ordering introspection."""

    def __init__(self):
        self.scene_calls = 0
        self.refine_order: list[int] = []

    def generate_scene_narrative(self, prompt: str, scene_packet):
        del prompt
        self.scene_calls += 1
        return {
            "narrative_paragraph": f"Scene {scene_packet.scene_id} summary.",
            "key_moments": [f"Scene {scene_packet.scene_id} moment."],
            "mentioned_entities": [item.name for item in scene_packet.entities[:2]],
            "mentioned_events": [item.event for item in scene_packet.events[:2]],
        }

    def refine_synopsis(self, current_synopsis: str, scene_narrative):
        self.refine_order.append(scene_narrative.scene_id)
        chunk = f"S{scene_narrative.scene_id}"
        if not current_synopsis:
            return chunk
        return f"{current_synopsis}->{chunk}"


def _frame(frame_id: int, timestamp: str, labels: list[str], faces: int = 0) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "files": {
            "original": f"jobs/job-1/frames/original/frame_{frame_id}.jpg",
            "segmentation": f"jobs/job-1/frames/seg/frame_{frame_id}.jpg",
            "detection": f"jobs/job-1/frames/det/frame_{frame_id}.jpg",
            "face": f"jobs/job-1/frames/face/frame_{frame_id}.jpg",
        },
        "analysis": {
            "semantic_segmentation": [],
            "object_detection": [
                {"track_id": f"{label}_{frame_id}_{index}", "label": label, "confidence": 0.9, "box": [0, 0, 1, 1]}
                for index, label in enumerate(labels, start=1)
            ],
            "face_recognition": [
                {"face_id": i + 1, "identity_id": f"face_{i + 1}", "confidence": 0.95, "coordinates": [0, 0, 1, 1]}
                for i in range(faces)
            ],
            "enrichment": {},
        },
        "analysis_artifacts": {
            "json": f"jobs/job-1/analysis/json/frame_{frame_id}.json",
        },
        "metadata": {
            "provenance": {
                "job_id": "job-1",
                "scene_id": None,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "source_artifact_key": f"jobs/job-1/frames/original/frame_{frame_id}.jpg",
            },
            "model_provenance": [],
            "evidence_anchors": [],
        },
    }


def test_generates_scene_outputs_and_summary(monkeypatch):
    store = FakeMediaStore()
    llm = RecordingLLM()
    monkeypatch.setattr("app.video_understanding.LANGGRAPH_AVAILABLE", False)
    monkeypatch.setattr("app.video_understanding.build_scene_llm_client", lambda _settings: llm)

    result = run_scene_understanding_pipeline(
        job_id="job-1",
        scenes=[(0.0, 4.0), (4.0, 8.0)],
        frame_results=[
            _frame(0, "00:00:01.000", ["person", "car"]),
            _frame(1, "00:00:05.000", ["dog"]),
        ],
        settings=_settings(),
        media_store=store,
    )

    assert llm.scene_calls == 2
    assert len(result["scene_narratives"]) == 2
    assert result["video_synopsis"] is not None
    assert result["video_synopsis"]["artifact"] == "jobs/job-1/summary/synopsis.json"
    uploaded_kinds = [kind for kind, _scene_id, _payload in store.scene_uploads]
    assert uploaded_kinds.count("packet") == 2
    assert uploaded_kinds.count("narrative") == 2


def test_keyframe_policy_respects_max_limit(monkeypatch):
    store = FakeMediaStore()
    llm = RecordingLLM()
    monkeypatch.setattr("app.video_understanding.LANGGRAPH_AVAILABLE", False)
    monkeypatch.setattr("app.video_understanding.build_scene_llm_client", lambda _settings: llm)

    result = run_scene_understanding_pipeline(
        job_id="job-1",
        scenes=[(0.0, 10.0)],
        frame_results=[
            _frame(0, "00:00:01.000", ["person", "car", "dog"]),
            _frame(1, "00:00:02.000", ["truck", "bike"]),
            _frame(2, "00:00:03.000", ["cat"]),
        ],
        settings=_settings(
            scene_packet_disambiguation_label_threshold=1,
            scene_packet_max_keyframes=2,
        ),
        media_store=store,
    )

    assert len(result["scene_narratives"]) == 1
    packet_payload = [payload for kind, _scene_id, payload in store.scene_uploads if kind == "packet"][0].decode("utf-8")
    packet = json.loads(packet_payload)
    assert len(packet["keyframes"]) == 2


def test_retries_on_faithfulness_failure(monkeypatch):
    class RetryLLM:
        def __init__(self):
            self.calls = 0

        def generate_scene_narrative(self, prompt: str, scene_packet):
            del prompt
            self.calls += 1
            if self.calls == 1:
                return {
                    "narrative_paragraph": "Bad response.",
                    "key_moments": ["bad"],
                    "mentioned_entities": ["unsupported_entity"],
                    "mentioned_events": [],
                }
            return {
                "narrative_paragraph": "Recovered response.",
                "key_moments": ["ok"],
                "mentioned_entities": [item.name for item in scene_packet.entities[:1]],
                "mentioned_events": [item.event for item in scene_packet.events[:1]],
            }

        def refine_synopsis(self, current_synopsis: str, scene_narrative):
            del scene_narrative
            return current_synopsis or "synopsis"

    llm = RetryLLM()
    store = FakeMediaStore()
    monkeypatch.setattr("app.video_understanding.LANGGRAPH_AVAILABLE", False)
    monkeypatch.setattr("app.video_understanding.build_scene_llm_client", lambda _settings: llm)

    result = run_scene_understanding_pipeline(
        job_id="job-1",
        scenes=[(0.0, 5.0)],
        frame_results=[_frame(0, "00:00:01.000", ["person"])],
        settings=_settings(scene_llm_retry_count=1),
        media_store=store,
    )

    assert llm.calls == 2
    assert result["scene_narratives"][0]["narrative_paragraph"] == "Recovered response."


def test_synopsis_refine_runs_in_chronological_order(monkeypatch):
    store = FakeMediaStore()
    llm = RecordingLLM()
    monkeypatch.setattr("app.video_understanding.LANGGRAPH_AVAILABLE", False)
    monkeypatch.setattr("app.video_understanding.build_scene_llm_client", lambda _settings: llm)

    run_scene_understanding_pipeline(
        job_id="job-1",
        scenes=[(10.0, 12.0), (0.0, 2.0)],
        frame_results=[
            _frame(0, "00:00:11.000", ["car"]),
            _frame(1, "00:00:01.000", ["person"]),
        ],
        settings=_settings(),
        media_store=store,
    )

    assert llm.refine_order == [1, 0]


def test_trace_metadata_propagates_when_enabled(monkeypatch):
    store = FakeMediaStore()
    llm = RecordingLLM()
    monkeypatch.setattr("app.video_understanding.LANGGRAPH_AVAILABLE", False)
    monkeypatch.setattr("app.video_understanding.build_scene_llm_client", lambda _settings: llm)

    result = run_scene_understanding_pipeline(
        job_id="job-1",
        scenes=[(0.0, 4.0)],
        frame_results=[_frame(0, "00:00:01.000", ["person"])],
        settings=_settings(langsmith_tracing_enabled=True, langsmith_project="dev-evals"),
        media_store=store,
    )

    narrative_trace = result["scene_narratives"][0]["trace"]
    synopsis_trace = result["video_synopsis"]["trace"]
    assert narrative_trace is not None
    assert synopsis_trace is not None
    assert narrative_trace["project"] == "dev-evals"
    assert narrative_trace["stage"] == "scene_narrative"
    assert synopsis_trace["stage"] == "video_synopsis"


def test_packet_builder_ignores_runtime_prompt_and_model_policy_changes():
    base_kwargs = {
        "job_id": "job-1",
        "scenes": [(0.0, 4.0)],
        "frame_results": [_frame(0, "00:00:01.000", ["person", "car"])],
    }
    store_v1 = FakeMediaStore()
    store_v2 = FakeMediaStore()

    packets_v1 = build_scene_packets(
        **base_kwargs,
        settings=_settings(
            scene_model_id="gemini-scene-v1",
            synopsis_model_id="gemini-synopsis-v1",
            scene_ai_prompt_version="v1",
        ),
        media_store=store_v1,
    )
    packets_v2 = build_scene_packets(
        **base_kwargs,
        settings=_settings(
            scene_model_id="gemini-scene-v2",
            synopsis_model_id="gemini-synopsis-v2",
            scene_ai_prompt_version="v2",
        ),
        media_store=store_v2,
    )

    assert [packet.packet_payload for packet in packets_v1] == [packet.packet_payload for packet in packets_v2]
    assert [packet.corpus_entities for packet in packets_v1] == [packet.corpus_entities for packet in packets_v2]
    assert [packet.corpus_events for packet in packets_v1] == [packet.corpus_events for packet in packets_v2]
    assert [packet.corpus_relations for packet in packets_v1] == [packet.corpus_relations for packet in packets_v2]
    assert [packet.retrieval_chunks for packet in packets_v1] == [packet.retrieval_chunks for packet in packets_v2]


def test_parse_scene_json_accepts_strict_json():
    payload = '{"narrative_paragraph":"Scene summary.","key_moments":["moment 1"],"mentioned_entities":[],"mentioned_events":[]}'
    parsed = GeminiSceneLLMClient._parse_scene_json(payload)
    assert parsed["narrative_paragraph"] == "Scene summary."
    assert parsed["key_moments"] == ["moment 1"]


def test_parse_scene_json_rejects_wrapped_response():
    payload = "Here is the JSON result:\n{}"
    with pytest.raises(ValueError):
        GeminiSceneLLMClient._parse_scene_json(payload)


def test_parse_scene_json_rejects_non_json():
    with pytest.raises(ValueError):
        GeminiSceneLLMClient._parse_scene_json("not-json")


def test_packet_payload_includes_multimodal_evidence_links():
    store = FakeMediaStore()
    packets = build_scene_packets(
        job_id="job-1",
        scenes=[(0.0, 4.0)],
        frame_results=[_frame(0, "00:00:01.000", ["person"], faces=1)],
        settings=_settings(scene_packet_disambiguation_label_threshold=1),
        media_store=store,
    )

    assert len(packets) == 1
    packet_payload = [payload for kind, _scene_id, payload in store.scene_uploads if kind == "packet"][0].decode("utf-8")
    parsed = json.loads(packet_payload)
    assert len(parsed["scene_frames"]) == 1

    modalities = {item["modality"] for item in parsed["scene_frames"][0]["modalities"]}
    assert modalities == {
        "original",
        "object_detection",
        "semantic_segmentation",
        "face_recognition",
    }

    evidence_ids = {item["evidence_id"] for item in parsed["evidence_refs"]}
    assert evidence_ids
    assert evidence_ids == set(parsed["evidence_index"].keys())


def test_face_modality_is_conditional_in_scene_packet():
    packets = build_scene_packets(
        job_id="job-1",
        scenes=[(0.0, 4.0)],
        frame_results=[_frame(0, "00:00:01.000", ["person"], faces=0)],
        settings=_settings(scene_packet_disambiguation_label_threshold=1),
        media_store=FakeMediaStore(),
    )

    packet = packets[0]
    assert packet.scene_frames[0].has_faces is False
    assert "face_recognition" not in packet.scene_frames[0].modality_image_keys
    assert all(item.modality != "face_recognition" for item in packet.evidence_refs)


def test_packet_validation_rejects_missing_required_modality():
    frame = _frame(0, "00:00:01.000", ["person"])
    frame["files"].pop("segmentation", None)

    with pytest.raises(ScenePacketValidationError):
        build_scene_packets(
            job_id="job-1",
            scenes=[(0.0, 4.0)],
            frame_results=[frame],
            settings=_settings(scene_packet_disambiguation_label_threshold=1),
            media_store=FakeMediaStore(),
        )


def test_multimodal_message_assembly_contains_evidence_links():
    packet = build_scene_packets(
        job_id="job-1",
        scenes=[(0.0, 4.0)],
        frame_results=[_frame(0, "00:00:01.000", ["person"], faces=1)],
        settings=_settings(scene_packet_disambiguation_label_threshold=1),
        media_store=FakeMediaStore(),
    )[0]
    prompt = build_scene_prompt(packet, policy=PromptPolicy(version="v2"))
    messages = build_scene_multimodal_messages(prompt, packet)

    assert len(messages) == 1
    content = messages[0]["content"]
    evidence_text_blocks = [
        item
        for item in content
        if item.get("type") == "text" and str(item.get("text", "")).startswith("EVIDENCE ")
    ]
    image_blocks = [item for item in content if item.get("type") == "image"]

    assert "Grounding contract:" in prompt
    assert "Prompt profile: v2." in prompt
    assert len(evidence_text_blocks) == len(packet.evidence_refs)
    assert len(image_blocks) == len(packet.evidence_refs)


def test_langgraph_state_carries_evidence_index(monkeypatch):
    store = FakeMediaStore()
    llm = RecordingLLM()

    class _FakeCompiledGraph:
        def __init__(self, nodes):
            self._nodes = nodes

        def invoke(self, state):
            current = dict(state)
            current.update(self._nodes["packets"](current))
            assert current["evidence_index"]
            current.update(self._nodes["narratives"](current))
            current.update(self._nodes["synopsis"](current))
            return current

    class _FakeStateGraph:
        def __init__(self, _state_type):
            self._nodes = {}

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def add_edge(self, _source, _target):
            return None

        def compile(self):
            return _FakeCompiledGraph(self._nodes)

    monkeypatch.setattr("app.video_understanding.LANGGRAPH_AVAILABLE", True)
    monkeypatch.setattr("app.video_understanding.StateGraph", _FakeStateGraph)
    monkeypatch.setattr("app.video_understanding.START", "start")
    monkeypatch.setattr("app.video_understanding.END", "end")
    monkeypatch.setattr("app.video_understanding.build_scene_llm_client", lambda _settings: llm)

    result = run_scene_understanding_pipeline(
        job_id="job-1",
        scenes=[(0.0, 4.0)],
        frame_results=[_frame(0, "00:00:01.000", ["person"], faces=1)],
        settings=_settings(),
        media_store=store,
    )

    assert len(result["scene_narratives"]) == 1
    assert result["video_synopsis"] is not None


def test_scene_packet_validation_errors_fail_fast_with_langgraph(monkeypatch):
    def _raise_validation_error(**_kwargs):
        raise ScenePacketValidationError("invalid scene packet")

    monkeypatch.setattr("app.video_understanding.LANGGRAPH_AVAILABLE", True)
    monkeypatch.setattr("app.video_understanding._run_with_langgraph", _raise_validation_error)
    monkeypatch.setattr("app.video_understanding.build_scene_llm_client", lambda _settings: RecordingLLM())

    with pytest.raises(ScenePacketValidationError):
        run_scene_understanding_pipeline(
            job_id="job-1",
            scenes=[(0.0, 4.0)],
            frame_results=[_frame(0, "00:00:01.000", ["person"])],
            settings=_settings(),
            media_store=FakeMediaStore(),
        )
