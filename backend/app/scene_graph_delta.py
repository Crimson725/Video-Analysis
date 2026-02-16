"""SceneGraphDelta Pydantic models, validation, repair, and normalization."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Evidence(BaseModel):
    """Grounding evidence linking LLM output to specific frames and tracks."""

    supporting_frames: list[int] = Field(
        ...,
        min_length=1,
        description="Frame IDs from the selected keyframes that support this assertion.",
    )
    supporting_track_ids: list[str] = Field(
        default_factory=list,
        description="Track IDs (object_track_id or video_person_id) supporting this assertion.",
    )


class EntityDelta(BaseModel):
    """LLM-enriched entity with semantic attributes grounded in CV evidence."""

    entity_id: str = Field(
        ...,
        description="Deterministic CV ID (video_person_id or object_track_id).",
    )
    type: str = Field(
        ...,
        description='Entity type, e.g. "Person", "Object".',
    )
    attributes: dict[str, object] = Field(
        default_factory=dict,
        description="Semantic enrichment attributes (clothing, emotion, color, etc.).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence in this entity enrichment.",
    )
    evidence: Evidence


class RelationDelta(BaseModel):
    """Semantic relation triple between two entities with temporal span."""

    subject_id: str = Field(..., description="Entity ID of the subject.")
    predicate: str = Field(..., description="Relation predicate from allowed predicates list.")
    object_id: str = Field(..., description="Entity ID of the object.")
    time_span_s: tuple[float, float] = Field(
        ...,
        description="Temporal span in seconds (start, end).",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence


class Participant(BaseModel):
    """Event participant with role annotation."""

    entity_id: str = Field(..., description="Entity ID of the participant.")
    role: str = Field(..., description='Role in the event, e.g. "giver", "receiver".')


class EventDelta(BaseModel):
    """Typed event with participants and temporal span."""

    event_id: str = Field(..., description="Deterministic event ID per scene.")
    event_type: str = Field(..., description='Event type, e.g. "handoff_object".')
    participants: list[Participant] = Field(default_factory=list)
    time_span_s: tuple[float, float] = Field(
        ...,
        description="Temporal span in seconds (start, end).",
    )
    summary: str = Field(..., description="Brief textual summary of the event.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence


class SceneGraphDelta(BaseModel):
    """Structured output schema for LLM scene enrichment."""

    scene_id: str = Field(..., description="Scene identifier.")
    entities: list[EntityDelta] = Field(default_factory=list)
    relations: list[RelationDelta] = Field(default_factory=list)
    events: list[EventDelta] = Field(default_factory=list)
    scene_summary: str | None = Field(
        default=None,
        description="Optional brief scene summary from the LLM.",
    )
    scene_tags: list[str] = Field(default_factory=list)
    open_questions: list[dict[str, object]] = Field(
        default_factory=list,
        description="LLM-flagged uncertainties about the scene.",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_delta(
    delta: SceneGraphDelta,
    *,
    allowed_predicates: list[str],
    selected_frame_ids: list[int],
    tracks_index_keys: set[str],
) -> list[str]:
    """Validate a SceneGraphDelta against schema, ontology, evidence, and ID constraints.

    Returns a list of error messages. Empty list means validation passed.
    """
    errors: list[str] = []
    allowed_set = set(allowed_predicates)
    frame_set = set(selected_frame_ids)

    # Validate entities
    for entity in delta.entities:
        # ID resolution
        if entity.entity_id not in tracks_index_keys:
            errors.append(
                f"Entity ID '{entity.entity_id}' not found in tracks_index"
            )
        # Evidence validity
        for fid in entity.evidence.supporting_frames:
            if fid not in frame_set:
                errors.append(
                    f"Evidence frame_id {fid} not in selected keyframes "
                    f"(entity '{entity.entity_id}')"
                )

    # Validate relations
    for rel in delta.relations:
        # Ontology check
        if rel.predicate not in allowed_set:
            errors.append(
                f"Predicate '{rel.predicate}' not in allowed_predicates"
            )
        # ID resolution
        if rel.subject_id not in tracks_index_keys:
            errors.append(
                f"Entity ID '{rel.subject_id}' not found in tracks_index "
                f"(relation subject)"
            )
        if rel.object_id not in tracks_index_keys:
            errors.append(
                f"Entity ID '{rel.object_id}' not found in tracks_index "
                f"(relation object)"
            )
        # Evidence validity
        for fid in rel.evidence.supporting_frames:
            if fid not in frame_set:
                errors.append(
                    f"Evidence frame_id {fid} not in selected keyframes "
                    f"(relation '{rel.subject_id}->{rel.predicate}->{rel.object_id}')"
                )

    # Validate events
    for event in delta.events:
        for participant in event.participants:
            if participant.entity_id not in tracks_index_keys:
                errors.append(
                    f"Participant entity_id '{participant.entity_id}' not found "
                    f"in tracks_index (event '{event.event_id}')"
                )
        for fid in event.evidence.supporting_frames:
            if fid not in frame_set:
                errors.append(
                    f"Evidence frame_id {fid} not in selected keyframes "
                    f"(event '{event.event_id}')"
                )

    return errors


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def repair_delta(
    delta: SceneGraphDelta,
    validation_errors: list[str],
    *,
    llm_client: Any,
    scene_packet_json: str,
    image_urls: list[str] | None = None,
    max_retries: int = 1,
) -> tuple[SceneGraphDelta | None, list[str]]:
    """Attempt to repair a failed delta by sending errors back to the LLM.

    Returns (repaired_delta, remaining_errors). If repair fails,
    returns (None, errors).
    """
    import json as json_mod

    from pydantic import ValidationError

    for attempt in range(max_retries):
        error_summary = "\n".join(f"- {e}" for e in validation_errors)
        repair_prompt = (
            "The following SceneGraphDelta had validation errors. "
            "Please fix them and return a corrected JSON.\n\n"
            f"Validation errors:\n{error_summary}\n\n"
            f"Original delta JSON:\n{delta.model_dump_json(indent=2)}\n\n"
            f"Scene context:\n{scene_packet_json}\n\n"
            "Return ONLY the corrected SceneGraphDelta JSON."
        )

        try:
            if hasattr(llm_client, "generate_scene_narrative"):
                # Use the same LLM interface — pass prompt and get raw dict/str back
                from app.scene_runtime_contracts import ScenePacket

                raw = llm_client.generate_scene_narrative(repair_prompt, None)
            elif hasattr(llm_client, "invoke"):
                response = llm_client.invoke(repair_prompt)
                if hasattr(response, "content"):
                    raw = response.content
                else:
                    raw = str(response)
            else:
                raw = str(llm_client(repair_prompt))

            if isinstance(raw, str):
                # Try to parse JSON from string
                text = raw.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    lines = [ln for ln in lines if not ln.strip().startswith("```")]
                    text = "\n".join(lines)
                raw = json_mod.loads(text)

            repaired = SceneGraphDelta.model_validate(raw)
            return repaired, []
        except (ValidationError, json_mod.JSONDecodeError, Exception) as exc:
            logger.warning(
                "Repair attempt %d failed: %s", attempt + 1, exc
            )
            validation_errors = [f"Repair attempt {attempt + 1} failed: {exc}"]

    return None, validation_errors


# ---------------------------------------------------------------------------
# spaCy Normalization
# ---------------------------------------------------------------------------

# Color canonicalization map
_COLOR_CANONICAL: dict[str, str] = {
    "dark blue": "navy",
    "darkblue": "navy",
    "light blue": "sky blue",
    "lightblue": "sky blue",
    "dark red": "maroon",
    "darkred": "maroon",
    "light red": "pink",
    "dark green": "forest green",
    "light green": "lime",
    "dark grey": "charcoal",
    "dark gray": "charcoal",
    "light grey": "silver",
    "light gray": "silver",
    "off-white": "ivory",
    "off white": "ivory",
}


def _normalize_color_text(text: str) -> str:
    """Standardize color descriptions in text."""
    lower = text.lower()
    for pattern, canonical in _COLOR_CANONICAL.items():
        if pattern in lower:
            lower = lower.replace(pattern, canonical)
    # Capitalize first letter of each sentence
    return lower


def _extract_ner_attributes(text: str, nlp: Any) -> dict[str, str]:
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    extracted: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ == "ORG":
            extracted["brand"] = ent.text
        elif ent.label_ == "GPE" or ent.label_ == "LOC":
            extracted["location"] = ent.text
        elif ent.label_ == "PERSON":
            extracted["person_name"] = ent.text
    return extracted


def normalize_delta(delta: SceneGraphDelta) -> SceneGraphDelta:
    """Apply spaCy normalization to text fields in a SceneGraphDelta.

    Normalization includes:
    - Clothing color standardization
    - NER extraction from OCR text
    - Attribute value canonicalization

    Does NOT alter entity IDs, predicates, or evidence references.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found; skipping NER normalization")
            nlp = None
    except ImportError:
        logger.warning("spaCy not installed; skipping normalization")
        nlp = None

    normalized_entities: list[EntityDelta] = []
    for entity in delta.entities:
        new_attrs = dict(entity.attributes)

        # Color normalization on clothing attributes
        for key in ("clothing", "color", "clothing_color"):
            if key in new_attrs:
                val = new_attrs[key]
                if isinstance(val, list):
                    new_attrs[key] = [_normalize_color_text(str(v)) for v in val]
                elif isinstance(val, str):
                    new_attrs[key] = _normalize_color_text(val)

        # NER extraction from OCR text
        if nlp is not None:
            ocr_text = new_attrs.get("ocr_text", "")
            if isinstance(ocr_text, str) and ocr_text.strip():
                ner_attrs = _extract_ner_attributes(ocr_text, nlp)
                for ner_key, ner_val in ner_attrs.items():
                    if ner_key not in new_attrs:
                        new_attrs[ner_key] = ner_val

        normalized_entities.append(
            EntityDelta(
                entity_id=entity.entity_id,
                type=entity.type,
                attributes=new_attrs,
                confidence=entity.confidence,
                evidence=entity.evidence,
            )
        )

    return SceneGraphDelta(
        scene_id=delta.scene_id,
        entities=normalized_entities,
        relations=delta.relations,  # predicates/evidence unchanged
        events=delta.events,  # unchanged
        scene_summary=delta.scene_summary,
        scene_tags=delta.scene_tags,
        open_questions=delta.open_questions,
    )
