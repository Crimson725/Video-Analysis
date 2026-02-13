"""Worker-local runtime modules for prompts, model routing, and output repair/retry."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from app.scene_packet_builder import validate_narrative_faithfulness
from app.scene_runtime_contracts import (
    SceneLLMClient,
    SceneNarrative,
    SceneNarrativeModel,
    ScenePacket,
)

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PromptPolicy:
    """Versioned prompt policy input for worker runtime execution."""

    version: str = "v1"


def build_scene_prompt(scene_packet: ScenePacket, *, policy: PromptPolicy | None = None) -> str:
    """Build JSON-first prompt for one-call-per-scene narrative generation."""
    selected_policy = policy or PromptPolicy()
    keyframe_note = "No keyframes attached."
    if scene_packet.keyframes:
        refs = ", ".join(k.uri for k in scene_packet.keyframes)
        keyframe_note = f"Supporting keyframes: {refs}"
    return (
        "You are given one scene packet in JSON format. "
        "Generate a faithful narrative for only this scene.\n\n"
        f"Prompt profile: {selected_policy.version}.\n"
        "Rules:\n"
        "- Use only entities/events present in the packet.\n"
        "- Return only valid JSON with no markdown or wrappers.\n"
        "- Return JSON with keys: narrative_paragraph, key_moments, mentioned_entities, mentioned_events.\n"
        "- key_moments must be a non-empty array of concise bullet strings.\n"
        "- Keep narrative_paragraph short and chronological.\n\n"
        f"{keyframe_note}\n\n"
        "```json\n"
        f"{scene_packet.packet_payload}"
        "```\n"
    )


class FallbackSceneLLMClient:
    """Deterministic no-network fallback used when Gemini runtime is unavailable."""

    def __init__(self, scene_model_id: str, synopsis_model_id: str) -> None:
        self.scene_model_id = scene_model_id
        self.synopsis_model_id = synopsis_model_id

    def generate_scene_narrative(self, prompt: str, scene_packet: ScenePacket) -> dict[str, Any]:
        del prompt
        if scene_packet.entities:
            top_entities = ", ".join(e.name for e in scene_packet.entities[:3])
        else:
            top_entities = "no dominant entities"
        paragraph = (
            f"From {scene_packet.start_sec:.2f}s to {scene_packet.end_sec:.2f}s, "
            f"the scene shows {top_entities} with {scene_packet.objects_total} object signals "
            f"and {scene_packet.faces_total} face detections."
        )
        moments = [
            f"Scene duration is {scene_packet.duration_sec:.2f} seconds.",
            f"Detected {scene_packet.unique_labels} unique labels.",
        ]
        if scene_packet.events:
            moments.append(f"Notable event: {scene_packet.events[0].event}.")
        return {
            "narrative_paragraph": paragraph,
            "key_moments": moments,
            "mentioned_entities": [e.name for e in scene_packet.entities[:3]],
            "mentioned_events": [e.event for e in scene_packet.events[:3]],
        }

    def refine_synopsis(
        self,
        current_synopsis: str,
        scene_narrative: SceneNarrative,
    ) -> str:
        chunk = (
            f"[{scene_narrative.start_sec:.2f}-{scene_narrative.end_sec:.2f}s] "
            f"{scene_narrative.narrative_paragraph}"
        )
        if not current_synopsis:
            return chunk
        return f"{current_synopsis} {chunk}"


class GeminiSceneLLMClient:
    """Gemini-backed scene and synopsis generation client."""

    def __init__(self, google_api_key: str, scene_model_id: str, synopsis_model_id: str) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        scene_response_schema = SceneNarrativeModel.model_json_schema()
        self._scene_model = ChatGoogleGenerativeAI(
            model=scene_model_id,
            google_api_key=google_api_key,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=scene_response_schema,
        )
        self._synopsis_model = ChatGoogleGenerativeAI(
            model=synopsis_model_id,
            google_api_key=google_api_key,
            temperature=0.1,
        )

    @staticmethod
    def _to_text(response: Any) -> str:
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                blocks: list[str] = []
                for block in content:
                    if isinstance(block, str):
                        blocks.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        blocks.append(str(block["text"]))
                return "\n".join(blocks)
        return str(response)

    @staticmethod
    def _parse_scene_json(text: str) -> dict[str, Any]:
        """Parse strict JSON object output from Gemini JSON mode."""
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Gemini scene narrative response is not a JSON object")

    def generate_scene_narrative(self, prompt: str, scene_packet: ScenePacket) -> dict[str, Any]:
        del scene_packet
        response = self._scene_model.invoke(prompt)
        text = self._to_text(response).strip()
        return self._parse_scene_json(text)

    def refine_synopsis(
        self,
        current_synopsis: str,
        scene_narrative: SceneNarrative,
    ) -> str:
        if current_synopsis:
            prompt = (
                "Refine the existing video synopsis while preserving chronology and voice.\n"
                f"Current synopsis:\n{current_synopsis}\n\n"
                f"Next scene ({scene_narrative.start_sec:.2f}-{scene_narrative.end_sec:.2f}s):\n"
                f"{scene_narrative.narrative_paragraph}\n"
                f"Key moments: {json.dumps(scene_narrative.key_moments)}\n\n"
                "Return only the updated synopsis paragraph."
            )
        else:
            prompt = (
                "Create an initial video synopsis paragraph from this first scene.\n"
                f"Scene ({scene_narrative.start_sec:.2f}-{scene_narrative.end_sec:.2f}s):\n"
                f"{scene_narrative.narrative_paragraph}\n"
                f"Key moments: {json.dumps(scene_narrative.key_moments)}\n\n"
                "Return only the synopsis paragraph."
            )
        response = self._synopsis_model.invoke(prompt)
        return self._to_text(response).strip()


def build_scene_llm_client(settings: "Settings") -> SceneLLMClient:
    """Build configured scene LLM client with graceful fallback behavior."""
    if settings.google_api_key:
        try:
            return GeminiSceneLLMClient(
                google_api_key=settings.google_api_key,
                scene_model_id=settings.scene_model_id,
                synopsis_model_id=settings.synopsis_model_id,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency path
            logger.warning("Gemini client unavailable; using fallback client: %s", exc)
    return FallbackSceneLLMClient(
        scene_model_id=settings.scene_model_id,
        synopsis_model_id=settings.synopsis_model_id,
    )


def generate_scene_narrative_with_retries(
    *,
    scene_packet: ScenePacket,
    llm_client: SceneLLMClient,
    retry_count: int,
    prompt_policy: PromptPolicy | None = None,
) -> SceneNarrativeModel:
    """Generate one scene narrative with validation and repair retries."""
    prompt = build_scene_prompt(scene_packet, policy=prompt_policy)
    retries = max(0, int(retry_count))
    last_error: Exception | None = None
    for _attempt in range(retries + 1):
        try:
            raw = llm_client.generate_scene_narrative(prompt, scene_packet)
            parsed = SceneNarrativeModel.model_validate(raw)
            validate_narrative_faithfulness(scene_packet, parsed)
            return parsed
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Scene narrative generation failed for scene {scene_packet.scene_id}"
    ) from last_error
