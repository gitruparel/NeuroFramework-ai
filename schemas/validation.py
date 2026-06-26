"""Pydantic schemas representing file and MRI-level structural validation reports."""

from typing import List
from pydantic import BaseModel, Field


class FileValidationReport(BaseModel):
    """Checks verifying basic file read status, existence, integrity and readability."""

    exists: bool = Field(..., description="True if the file or folder exists.")
    readable: bool = Field(..., description="True if the platform has correct read permissions.")
    header_valid: bool = Field(..., description="True if file headers are readable and not corrupt.")
    corrupt: bool = Field(..., description="True if any format corruption errors are encountered.")


class MRIValidationReport(BaseModel):
    """Checks verifying internal voxel, dimension, intensity, and spacing properties."""

    voxel_spacing_valid: bool = Field(..., description="Checks if voxel spacing is within expected physics range.")
    dimensions_valid: bool = Field(..., description="Checks if dimensions match 3D/2D requirements.")
    intensity_valid: bool = Field(..., description="Checks for abnormal negative or flat zero intensities.")
    orientation_valid: bool = Field(..., description="Checks if spatial coordinate matrix orientation is standard.")
    metadata_complete: bool = Field(..., description="Checks if crucial patient/scanner labels are filled.")
    empty_slices_detected: bool = Field(..., description="Checks for blank zero-intensity slices in 3D volume.")


class ValidationReport(BaseModel):
    """Aggregated double-layer validation container for file and MRI checks."""

    file_validation: FileValidationReport = Field(..., description="File level check details.")
    mri_validation: MRIValidationReport = Field(..., description="MRI/Volume level check details.")
    is_valid: bool = Field(..., description="True if all file and MRI validation checks pass successfully.")
    errors: List[str] = Field(default_factory=list, description="List of descriptive failure messages.")
