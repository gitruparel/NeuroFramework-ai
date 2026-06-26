"""Pydantic schemas representing structural MRI scan quality assessments."""

from typing import Dict, Any
from pydantic import BaseModel, Field


class QualityReport(BaseModel):
    """Visual noise, blur, artifact, and resolution indicators calculated on structural scans."""

    noise_score: float = Field(..., description="Estimated signal-to-noise ratio or noise metric.")
    blur_score: float = Field(..., description="Estimated blur/fuzziness index metric.")
    motion_score: float = Field(..., description="Estimated motion/ringing artifacts presence index.")
    contrast_score: float = Field(..., description="Intensity contrast between tissues (e.g., white matter/gray matter ratio).")
    dynamic_range: float = Field(..., description="Calculated dynamic range of voxel values.")
    resolution: float = Field(..., description="Voxel volume spatial resolution score.")
    slice_count: int = Field(..., description="Total slice count in the 3D volume.")
    missing_slices: bool = Field(default=False, description="Flag indicating if intermediate slice checks detected missing/dropped data.")
    has_artifacts: bool = Field(default=False, description="Flag indicating overall presence of severe scanning artifacts.")
    overall_score: float = Field(..., description="Overall normalized scan quality check rating [0.0 - 1.0].")
    extra_metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional custom quality assessment parameters.")
