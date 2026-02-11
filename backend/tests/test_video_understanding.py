"""Unit tests for app.video_understanding."""

from types import SimpleNamespace

import pytest

from app.storage import build_scene_key, build_summary_key
from app.video_understanding import GeminiSceneLLMClient, run_scene_understanding_pipeline


def _settings(**overrides):
    defaults = {
        "google_api_key": "",
        "scene_model_id": "gemini-2.5-flash-lite",
        "synopsis_model_id": "gemini-2.5-flash-lite",
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
            "object_detection": [{"label": label, "confidence": 0.9, "box": [0, 0, 1, 1]} for label in labels],
            "face_recognition": [{"face_id": i + 1, "confidence": 0.95, "coordinates": [0, 0, 1, 1]} for i in range(faces)],
        },
        "analysis_artifacts": {
            "json": f"jobs/job-1/analysis/json/frame_{frame_id}.json",
            "toon": f"jobs/job-1/analysis/toon/frame_{frame_id}.toon",
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
    assert "keyframes[2]{frameId,timestamp,uri}:" in packet_payload


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


def test_parse_scene_json_accepts_fenced_block():
    payload = """
    ```json
    {"narrative_paragraph":"Scene summary.","key_moments":["moment 1"],"mentioned_entities":[],"mentioned_events":[]}
    ```
    """
    parsed = GeminiSceneLLMClient._parse_scene_json(payload)
    assert parsed["narrative_paragraph"] == "Scene summary."
    assert parsed["key_moments"] == ["moment 1"]


def test_parse_scene_json_accepts_wrapped_response():
    payload = (
        "Here is the JSON result:\n"
        '{"narrative_paragraph":"Scene summary.","key_moments":["moment 1"],'
        '"mentioned_entities":[],"mentioned_events":[]}'
    )
    parsed = GeminiSceneLLMClient._parse_scene_json(payload)
    assert parsed["narrative_paragraph"] == "Scene summary."
    assert parsed["key_moments"] == ["moment 1"]


def test_parse_scene_json_rejects_non_json():
    with pytest.raises(ValueError):
        GeminiSceneLLMClient._parse_scene_json("not-json")
