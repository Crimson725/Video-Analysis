"""Pydantic models for video analysis API responses."""

from pydantic import BaseModel, ConfigDict, Field


class SegmentationItem(BaseModel):
    """Single segmentation mask result."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    object_id: int
    class_name: str = Field(..., alias="class")
    mask_polygon: list[list[int]]


class DetectionItem(BaseModel):
    """Single object detection result."""

    label: str
    confidence: float
    box: list[int] = Field(..., min_length=4, max_length=4)


class FaceItem(BaseModel):
    """Single face recognition result."""

    face_id: int
    identity_id: str | None = None
    confidence: float
    coordinates: list[int] = Field(..., min_length=4, max_length=4)


class FrameFiles(BaseModel):
    """Signed URLs to generated frame images stored in object storage."""

    original: str
    segmentation: str
    detection: str
    face: str


class FrameAnalysis(BaseModel):
    """Analysis results for a single frame."""

    semantic_segmentation: list[SegmentationItem] = []
    object_detection: list[DetectionItem] = []
    face_recognition: list[FaceItem] = []


class AnalysisArtifacts(BaseModel):
    """R2-backed per-frame analysis artifacts."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    json_artifact: str = Field(..., alias="json")
    toon: str


class FrameResult(BaseModel):
    """Complete result for a single extracted frame."""

    frame_id: int
    timestamp: str
    files: FrameFiles
    analysis: FrameAnalysis
    analysis_artifacts: AnalysisArtifacts


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
    trace: dict[str, str] | None = None


class VideoSynopsisResult(BaseModel):
    """Generated full-video synopsis output."""

    synopsis: str
    artifact: str
    model: str
    trace: dict[str, str] | None = None


class JobResult(BaseModel):
    """Full analysis result for a job."""

    job_id: str
    frames: list[FrameResult]
    scene_narratives: list[SceneNarrativeResult] = Field(default_factory=list)
    video_synopsis: VideoSynopsisResult | None = None


class JobStatus(BaseModel):
    """Job status response."""

    job_id: str
    status: str
    error: str | None = None
