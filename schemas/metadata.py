"""Pydantic schemas representing granular and wrapper medical image metadata."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    """Metadata detailing structural scan dimensions, orientations, and resolution spacing."""

    voxel_dims: List[float] = Field(..., description="Voxel spacing/resolution spacing parameters [dx, dy, dz].")
    dimensions: List[int] = Field(..., description="Voxel dimension resolution counts [x, y, z].")
    orientation: str | None = Field(default=None, description="Orientation sequence code (e.g. RAS, LAS).")
    modality: str = Field(default="T1w", description="Imaging modality (e.g. T1w, T2w, FLAIR).")


class ScannerMetadata(BaseModel):
    """Metadata detailing MRI acquisition hardware settings."""

    manufacturer: str | None = Field(default=None, description="Scanner manufacturer name.")
    model: str | None = Field(default=None, description="Scanner model identification.")
    magnetic_field_strength: float | None = Field(default=None, description="Scanner field strength in Tesla.")
    scan_date: str | None = Field(default=None, description="Timestamp indicating scan acquisition date.")


class PatientMetadata(BaseModel):
    """Metadata detailing patient parameters anonymized for privacy."""

    patient_id: str = Field(..., description="Anonymized unique patient identifier.")
    age: float | None = Field(default=None, description="Calculated patient age at scan time.")
    gender: str | None = Field(default=None, description="Patient sex classification.")


class MRIMetadata(BaseModel):
    """Unified wrapper combining patient, scanner, and image metadata."""

    image: ImageMetadata = Field(..., description="Image structural properties.")
    scanner: ScannerMetadata = Field(default_factory=ScannerMetadata, description="Acquisition scanner properties.")
    patient: PatientMetadata = Field(..., description="Anonymized patient properties.")
