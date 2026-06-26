"""Schemas representing model predictions."""

from typing import Dict, List
from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    """Represents a single classification class probability."""

    class_name: str = Field(..., description="Target class classification name (e.g. Alzheimer, Autism).")
    probability: float = Field(..., description="Calculated probability [0.0 - 1.0].")


class Prediction(BaseModel):
    """Comprehensive container for model inference outputs, supporting segmentation or classifications."""

    patient_id: str = Field(..., description="Target patient identifier.")
    model_name: str = Field(..., description="Model version run for this prediction.")
    probabilities: List[ClassProbability] = Field(default_factory=list, description="Classification prediction classes.")
    segmentation_path: str | None = Field(default=None, description="Path where predicted voxel labels are stored.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Calculated inference-level metrics.")
