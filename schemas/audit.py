"""Pydantic schemas representing dataset auditing results."""

from typing import Dict, List
from pydantic import BaseModel, Field


class VoxelSpacingDistribution(BaseModel):
    """Aggregated spacing statistics across dimensions."""

    min_spacing: List[float] = Field(..., description="Minimum voxel spacing found [dx, dy, dz].")
    max_spacing: List[float] = Field(..., description="Maximum voxel spacing found [dx, dy, dz].")
    mean_spacing: List[float] = Field(..., description="Mean voxel spacing calculated across dataset.")


class AuditReport(BaseModel):
    """Packaging dataset-wide structural audit details."""

    dataset_path: str = Field(..., description="Root path scanned.")
    total_subjects: int = Field(..., description="Total unique subjects identified.")
    total_files: int = Field(..., description="Total file items scanned.")
    formats_present: List[str] = Field(default_factory=list, description="Unique formats found (nifti, dicom, image).")
    corrupt_files: List[str] = Field(default_factory=list, description="Paths of corrupt/unreadable files.")
    missing_files: List[str] = Field(default_factory=list, description="Paths of expected but missing scans.")
    voxel_spacings: VoxelSpacingDistribution | None = Field(default=None, description="Voxel spacing statistics.")
    unique_dimensions: List[List[int]] = Field(default_factory=list, description="Set of unique shapes found.")
    class_balance: Dict[str, int] = Field(default_factory=dict, description="Distribution count of classes found.")
    missing_metadata_count: int = Field(..., description="Count of loaded volumes missing crucial metadata.")
    scanner_manufacturers: Dict[str, int] = Field(default_factory=dict, description="Distribution of scanner manufacturers.")
