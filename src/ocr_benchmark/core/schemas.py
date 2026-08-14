from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    SKIPPED_NOT_CONFIGURED = "SKIPPED_NOT_CONFIGURED"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    EXCEPTION = "EXCEPTION"


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str
    label_count: int = 1
    label_type: str = "default"
    tags: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)
    critical_fields: list[str] = Field(default_factory=list)
    full_text: str = ""
    barcode_type: Optional[str] = None
    notes: str = ""


class Detection(BaseModel):
    text: str
    bbox: Optional[list[float]] = None
    confidence: Optional[float] = None


class Timing(BaseModel):
    preprocess_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    postprocess_ms: Optional[float] = None
    worker_startup_ms: Optional[float] = None
    model_load_ms: Optional[float] = None
    request_round_trip_ms: Optional[float] = None
    pipeline_e2e_ms: Optional[float] = None


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    image: str
    status: RunStatus = RunStatus.SUCCESS
    raw_text: str = ""
    detections: list[Detection] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    timing: Timing = Field(default_factory=Timing)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class Dataset(BaseModel):
    samples: list[GroundTruth]

    def paths(self, root: Path) -> list[Path]:
        return [root / sample.image for sample in self.samples]
