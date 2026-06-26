"""Metadata extractor implementation from RawMRI records."""

from typing import Any, List
import numpy as np
from schemas.metadata import ImageMetadata, MRIMetadata, PatientMetadata, ScannerMetadata
from schemas.mri import RawMRI


class MetadataExtractor:
    """Parses raw loaded scan headers and tensors to compile structured metadata."""

    def extract(self, raw: RawMRI) -> MRIMetadata:
        """Extracts and normalizes Image, Scanner, and Patient metadata from RawMRI."""
        # 1. Parse Image Metadata
        tensor_shape = list(raw.tensor.shape)
        voxel_dims = self._calculate_voxel_spacing(raw)
        
        # Determine orientation code (RAS coordinate style)
        orientation = "RAS" if raw.format != "image" else "2D"
        
        image_meta = ImageMetadata(
            voxel_dims=voxel_dims,
            dimensions=tensor_shape,
            orientation=orientation,
            modality="T1w" if raw.format != "image" else "Standard 2D",
        )

        # 2. Parse Scanner Metadata
        scanner_meta = self._extract_scanner_meta(raw)

        # 3. Parse Patient Metadata
        patient_meta = self._extract_patient_meta(raw)

        return MRIMetadata(
            image=image_meta,
            scanner=scanner_meta,
            patient=patient_meta,
        )

    def _calculate_voxel_spacing(self, raw: RawMRI) -> List[float]:
        """Calculates spacing dimensions from affine transformations or pixdim headers."""
        if raw.format == "image":
            return [1.0, 1.0, 1.0]

        # Extract scaling diagonal components from affine transformation matrix
        try:
            spacing = []
            for i in range(3):
                col_len = float(np.linalg.norm(raw.affine[:3, i]))
                spacing.append(col_len if col_len > 0 else 1.0)
            return spacing
        except Exception:
            # Fallback to header attributes if affine norm fails
            return [1.0, 1.0, 1.0]

    def _extract_scanner_meta(self, raw: RawMRI) -> ScannerMetadata:
        """Extracts scanner configuration parameters from DICOM/NIfTI fields."""
        if raw.format == "dicom":
            # Search typical DICOM tags in the header dictionary
            return ScannerMetadata(
                manufacturer=raw.header.get("0008|0070", "Unknown").strip(),
                model=raw.header.get("0008|1090", "Unknown").strip(),
                magnetic_field_strength=self._parse_float(raw.header.get("0018|0087")),
                scan_date=raw.header.get("0008|0020", None),
            )
        
        # NIfTI / Image format default placeholders
        return ScannerMetadata()

    def _extract_patient_meta(self, raw: RawMRI) -> PatientMetadata:
        """Extracts patient information from DICOM/NIfTI fields."""
        if raw.format == "dicom":
            return PatientMetadata(
                patient_id=raw.header.get("0010|0020", "Anonymous").strip(),
                age=self._parse_age(raw.header.get("0010|1010")),
                gender=raw.header.get("0010|0040", "Unknown").strip(),
            )
        
        return PatientMetadata(patient_id="Anonymous_Patient")

    def _parse_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except ValueError:
            return None

    def _parse_age(self, value: Any) -> float | None:
        if value is None:
            return None
        # DICOM age is often formatted as '030Y'
        val_str = str(value).strip()
        if val_str.endswith("Y") and val_str[:-1].isdigit():
            return float(val_str[:-1])
        try:
            return float(val_str)
        except ValueError:
            return None
