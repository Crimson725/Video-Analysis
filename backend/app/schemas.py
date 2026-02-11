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
    confidence: float
    coordinates: list[int] = Field(..., min_length=4, max_length=4)


class FrameFiles(BaseModel):
    """Paths to generated frame images."""

    original: str
    segmentation: str
    detection: str
    face: str


class FrameAnalysis(BaseModel):
    """Analysis results for a single frame."""

    semantic_segmentation: list[SegmentationItem] = []
    object_detection: list[DetectionItem] = []
    face_recognition: list[FaceItem] = []


class FrameResult(BaseModel):
    """Complete result for a single extracted frame."""

    frame_id: int
    timestamp: str
    files: FrameFiles
    analysis: FrameAnalysis


class JobResult(BaseModel):
    """Full analysis result for a job."""

    job_id: str
    frames: list[FrameResult]


class JobStatus(BaseModel):
    """Job status response."""

    job_id: str
    status: str
    error: str | None = None
