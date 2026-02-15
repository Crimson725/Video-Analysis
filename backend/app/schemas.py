"""Pydantic models for video analysis API responses and corpus artifacts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceAnchor(BaseModel):
    """Grounding reference to a source frame region or artifact span."""

    frame_id: int
    timestamp: str
    artifact_key: str
    bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    text_span: str | None = None


class ModelProvenance(BaseModel):
    """Model and configuration provenance attached to generated outputs."""

    component: str
    model_id: str
    model_version: str
    threshold: float | None = None


class SegmentationItem(BaseModel):
    """Single segmentation mask result."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    object_id: int
    class_name: str = Field(..., alias="class")
    mask_polygon: list[list[int]]
    palette_rgb: list[int] | None = Field(default=None, min_length=3, max_length=3)
    bbox_rgb: list[int] | None = Field(default=None, min_length=3, max_length=3)


class DetectionItem(BaseModel):
    """Single object detection result with stable tracking identifier."""

    track_id: str
    label: str
    confidence: float
    box: list[int] = Field(..., min_length=4, max_length=4)
    palette_rgb: list[int] | None = Field(default=None, min_length=3, max_length=3)
    bbox_rgb: list[int] | None = Field(default=None, min_length=3, max_length=3)
    person_track_id: str | None = None
    person_identity_id: str | None = None
    person_identity_source: str | None = None
    person_identity_confidence: float | None = None


class FaceItem(BaseModel):
    """Single face recognition result."""

    face_id: int
    identity_id: str
    confidence: float
    coordinates: list[int] = Field(..., min_length=4, max_length=4)
    palette_rgb: list[int] | None = Field(default=None, min_length=3, max_length=3)
    bbox_rgb: list[int] | None = Field(default=None, min_length=3, max_length=3)
    scene_person_id: str | None = None
    video_person_id: str | None = None
    match_confidence: float | None = None
    is_identity_ambiguous: bool | None = None
    embedding_model_id: str | None = None


class OCRBlock(BaseModel):
    """OCR enrichment block attached to a frame."""

    text: str
    confidence: float
    bbox: list[int] = Field(..., min_length=4, max_length=4)


class ActionLabel(BaseModel):
    """Action/activity enrichment label for a frame."""

    label: str
    confidence: float


class PoseKeypoint(BaseModel):
    """Single pose keypoint in pixel space."""

    x: float
    y: float
    confidence: float


class PoseItem(BaseModel):
    """Pose enrichment payload for a tracked subject."""

    track_id: str
    confidence: float
    keypoints: list[PoseKeypoint] = Field(default_factory=list)


class CameraMotionTag(BaseModel):
    """Camera or shot-level motion descriptor."""

    label: str
    confidence: float


class QualityFlags(BaseModel):
    """Frame quality flags used for corpus trust signals."""

    blur_score: float | None = None
    is_blurry: bool | None = None
    is_occluded: bool | None = None


class FrameEnrichment(BaseModel):
    """Optional CV enrichment outputs normalized for corpus use."""

    ocr_blocks: list[OCRBlock] = Field(default_factory=list)
    actions: list[ActionLabel] = Field(default_factory=list)
    poses: list[PoseItem] = Field(default_factory=list)
    camera_motion: CameraMotionTag | None = None
    quality: QualityFlags | None = None


class FrameProvenance(BaseModel):
    """Per-frame provenance used for corpus grounding."""

    job_id: str
    scene_id: int | None = None
    frame_id: int
    timestamp: str
    source_artifact_key: str


class FrameMetadata(BaseModel):
    """Metadata envelope for corpus grounding and traceability."""

    provenance: FrameProvenance
    model_provenance: list[ModelProvenance] = Field(default_factory=list)
    evidence_anchors: list[EvidenceAnchor] = Field(default_factory=list)


class FrameFiles(BaseModel):
    """Signed URLs to generated frame images stored in object storage."""

    original: str
    segmentation: str
    detection: str
    face: str


class FrameAnalysis(BaseModel):
    """Analysis results for a single frame."""

    semantic_segmentation: list[SegmentationItem] = Field(default_factory=list)
    object_detection: list[DetectionItem] = Field(default_factory=list)
    face_recognition: list[FaceItem] = Field(default_factory=list)
    enrichment: FrameEnrichment = Field(default_factory=FrameEnrichment)


class AnalysisArtifacts(BaseModel):
    """R2-backed per-frame analysis artifacts."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    json_artifact: str = Field(..., alias="json")


class FrameResult(BaseModel):
    """Complete result for a single extracted frame."""

    frame_id: int
    timestamp: str
    files: FrameFiles
    analysis: FrameAnalysis
    analysis_artifacts: AnalysisArtifacts
    metadata: FrameMetadata


class SceneTemporalSpan(BaseModel):
    """Temporal span metadata for corpus entities and events."""

    first_seen: float
    last_seen: float
    duration_sec: float


class SceneEntity(BaseModel):
    """Scene-level corpus-ready entity aggregate."""

    entity_id: str
    label: str
    entity_type: str
    count: int
    confidence: float
    temporal_span: SceneTemporalSpan
    evidence: list[EvidenceAnchor] = Field(min_length=1)
    track_id: str | None = None
    identity_id: str | None = None


class SceneEvent(BaseModel):
    """Scene-level corpus-ready event aggregate."""

    event_id: str
    event_type: str
    count: int
    confidence: float
    temporal_span: SceneTemporalSpan
    evidence: list[EvidenceAnchor] = Field(min_length=1)


class SceneRelation(BaseModel):
    """Scene-level corpus-ready relation aggregate."""

    relation_id: str
    source_entity_id: str
    target_entity_id: str
    predicate: str
    confidence: float
    temporal_span: SceneTemporalSpan
    evidence: list[EvidenceAnchor] = Field(min_length=1)


class RetrievalChunk(BaseModel):
    """RAG-ready chunk generated from scene understanding outputs."""

    chunk_id: str
    text: str
    source_entity_ids: list[str] = Field(default_factory=list)
    artifact_keys: list[str] = Field(default_factory=list)
    temporal_span: SceneTemporalSpan


class SceneCorpusArtifacts(BaseModel):
    """Deterministic references to scene-level corpus sources."""

    graph_bundle: str
    retrieval_bundle: str


class SceneCorpusPacket(BaseModel):
    """Corpus payload emitted at scene layer before job-level build."""

    entities: list[SceneEntity] = Field(default_factory=list)
    events: list[SceneEvent] = Field(default_factory=list)
    relations: list[SceneRelation] = Field(default_factory=list)
    retrieval_chunks: list[RetrievalChunk] = Field(default_factory=list)
    artifacts: SceneCorpusArtifacts


class SceneArtifacts(BaseModel):
    """R2-backed scene-level artifacts."""

    packet: str
    narrative: str


class SceneNarrativeResult(BaseModel):
    """Generated scene-level narrative output."""

    scene_id: int
    start_sec: float
    end_sec: float
    narrative_paragraph: str
    key_moments: list[str] = Field(default_factory=list, min_length=1)
    artifacts: SceneArtifacts
    corpus: SceneCorpusPacket | None = None
    trace: dict[str, str] | None = None


class VideoSynopsisResult(BaseModel):
    """Generated full-video synopsis output."""

    synopsis: str
    artifact: str
    model: str
    trace: dict[str, str] | None = None


class RetrievalChunkRecord(BaseModel):
    """Vector-search chunk with metadata references."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCorpusBundle(BaseModel):
    """Retrieval corpus bundle."""

    job_id: str
    chunks: list[RetrievalChunkRecord] = Field(default_factory=list)


class CorpusArtifacts(BaseModel):
    """Deterministic stored artifact references for corpus outputs."""

    retrieval_bundle: str


class JobCorpusResult(BaseModel):
    """Job-level corpus products emitted after scene understanding."""

    retrieval: RetrievalCorpusBundle
    artifacts: CorpusArtifacts
    ingest: dict[str, Any] | None = None


class JobResult(BaseModel):
    """Full analysis result for a job."""

    job_id: str
    frames: list[FrameResult]
    scene_narratives: list[SceneNarrativeResult] = Field(default_factory=list)
    video_synopsis: VideoSynopsisResult | None = None
    corpus: JobCorpusResult | None = None
    video_face_identities: dict[str, Any] | None = None
    video_person_tracks: dict[str, Any] | None = None


class JobStatus(BaseModel):
    """Job status response."""

    job_id: str
    status: str
    error: str | None = None
