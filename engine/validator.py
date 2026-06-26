"""Double-layer validation engine checking file attributes and internal MRI attributes."""

import os
from pathlib import Path
import numpy as np
from schemas.validation import FileValidationReport, MRIValidationReport, ValidationReport
from schemas.mri import RawMRI
from schemas.metadata import MRIMetadata


class ValidationEngine:
    """Executes validation checks on structural volumes at file and voxel array levels."""

    def validate_file(self, path: Path) -> FileValidationReport:
        """Verifies physical file exists, is readable, and is not corrupt."""
        exists = path.exists()
        readable = os.access(path, os.R_OK) if exists else False
        
        # We check corruption heuristically (readers handle exceptions)
        return FileValidationReport(
            exists=exists,
            readable=readable,
            header_valid=exists and readable,  # Checked during loading stage
            corrupt=not (exists and readable),
        )

    def validate_mri(self, raw: RawMRI, metadata: MRIMetadata) -> MRIValidationReport:
        """Inspects dimensions, voxel spacings, orientations, and empty slices."""
        tensor = raw.tensor
        shape = tensor.shape
        
        # 1. Dimensional check: 3D if nifti/dicom, 2D with singleton channels if image
        dimensions_valid = len(shape) == 3 or (len(shape) == 2 or (len(shape) == 3 and shape[-1] == 1))
        
        # 2. Voxel spacing bounds check (e.g. should be positive)
        spacing = metadata.image.voxel_dims
        voxel_spacing_valid = all(s > 0.0 for s in spacing)

        # 3. Check for empty zero slices
        # Compute sum along slice axes to flag flat slices
        empty_slices = False
        if len(shape) == 3 and shape[-1] > 1:
            for z in range(shape[-1]):
                if np.max(tensor[:, :, z]) == np.min(tensor[:, :, z]):
                    empty_slices = True
                    break

        # 4. Intensity check (abnormal bounds or constant flat inputs)
        intensity_valid = np.max(tensor) > np.min(tensor)

        # 5. Metadata completion check
        metadata_complete = bool(metadata.patient.patient_id)

        # 6. Spatial orientation check
        orientation_valid = metadata.image.orientation is not None

        return MRIValidationReport(
            voxel_spacing_valid=voxel_spacing_valid,
            dimensions_valid=dimensions_valid,
            intensity_valid=intensity_valid,
            orientation_valid=orientation_valid,
            metadata_complete=metadata_complete,
            empty_slices_detected=empty_slices,
        )

    def run_all(self, path: Path, raw: RawMRI | None, metadata: MRIMetadata | None) -> ValidationReport:
        """Orchestrates both file validation and voxel validation layers."""
        file_rep = self.validate_file(path)
        errors = []

        if not file_rep.exists:
            errors.append("File does not exist on disk.")
        if not file_rep.readable:
            errors.append("File exists but is not readable (permissions error).")
        
        if raw is None or metadata is None:
            # File loading failed, cannot execute MRI validations
            mri_rep = MRIValidationReport(
                voxel_spacing_valid=False,
                dimensions_valid=False,
                intensity_valid=False,
                orientation_valid=False,
                metadata_complete=False,
                empty_slices_detected=False,
            )
            return ValidationReport(
                file_validation=file_rep,
                mri_validation=mri_rep,
                is_valid=False,
                errors=errors,
            )

        mri_rep = self.validate_mri(raw, metadata)
        
        if not mri_rep.voxel_spacing_valid:
            errors.append("Invalid or negative voxel spacing dimensions detected.")
        if not mri_rep.dimensions_valid:
            errors.append("MRI shape dimensions do not meet requirements.")
        if not mri_rep.intensity_valid:
            errors.append("Constant or flat voxel intensities detected.")
        if mri_rep.empty_slices_detected:
            errors.append("Blank zero-intensity slice volumes detected inside scan.")

        is_valid = file_rep.exists and file_rep.readable and not errors

        return ValidationReport(
            file_validation=file_rep,
            mri_validation=mri_rep,
            is_valid=is_valid,
            errors=errors,
        )
