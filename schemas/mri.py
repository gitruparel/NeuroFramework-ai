"""Pydantic schemas representing RawMRI, preview images, statistics, and final MRIData packaging."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from schemas.metadata import MRIMetadata
from schemas.quality import QualityReport
from schemas.validation import ValidationReport
from schemas.processing import ProcessingRecord


class RawMRI(BaseModel):
    """Contains loaded raw medical volume attributes without secondary metadata or validations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensor: np.ndarray = Field(..., description="N-dimensional numpy array representing image voxel intensities.")
    affine: np.ndarray = Field(..., description="4x4 affine coordinate transformation matrix mapping voxels to patient space.")
    header: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary extracted directly from file headers.")
    format: str = Field(..., description="Detected image format: nifti, dicom, image.")
    source_path: Path = Field(..., description="Source path where the file was read.")


class ScanStatistics(BaseModel):
    """Statistical summary metrics of the image tensor voxel intensities."""

    min: float = Field(..., description="Minimum intensity value.")
    max: float = Field(..., description="Maximum intensity value.")
    mean: float = Field(..., description="Average intensity value.")
    std: float = Field(..., description="Standard deviation of intensity values.")
    shape: List[int] = Field(..., description="Tensor volume dimensions shape.")
    dtype: str = Field(..., description="Voxel intensity numerical data type.")


class PreviewImage(BaseModel):
    """Represents a generated orthogonal 2D slice preview."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray = Field(..., description="2D slice array scaled to grayscale [0-255].")
    plane: str = Field(..., description="Orthogonal slice plane (axial, coronal, sagittal).")
    slice_idx: int = Field(..., description="Voxel index of slice along plane.")
    title: str = Field(..., description="Title or header descriptor for visualization.")


class MRIData(BaseModel):
    """Unified packaged container representing the loaded structural MRI scan and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw: RawMRI = Field(..., description="Raw loaded volume and headers.")
    image: Optional[np.ndarray] = Field(default=None, description="Processed voxel intensity array.")
    affine: Optional[np.ndarray] = Field(default=None, description="Current affine coordinate transformation matrix mapping voxels to patient space.")
    brain_mask: Optional[np.ndarray] = Field(default=None, description="Binary skull-stripped brain mask tensor.")
    metadata: MRIMetadata = Field(..., description="Normalized patient, scanner, and acquisition metadata.")
    quality: QualityReport = Field(..., description="Automated scan quality check report.")
    validation: ValidationReport = Field(..., description="Double-layered file and MRI data validation checks.")
    history: List[ProcessingRecord | str] = Field(default_factory=list, description="Pipeline preprocessing step execution log history.")
    preview: List[PreviewImage] = Field(default_factory=list, description="Extracted axial, coronal, and sagittal orthogonal slice previews.")
    statistics: ScanStatistics = Field(..., description="Calculated voxel intensity statistics.")
