"""Volume conversion utilities (e.g. DICOM to NIfTI)."""

from pathlib import Path
from core.exceptions import MRIProcessingError


class MRIConverter:
    """Handles format transpositions (e.g. converting DICOM folders to NIfTI)."""

    def dicom_to_nifti(self, dicom_dir: Path | str, output_path: Path | str) -> None:
        """Skeleton converting DICOM series directory into single NIfTI file."""
        try:
            # Placeholder for conversion logic (e.g. using SimpleITK)
            pass
        except Exception as e:
            raise MRIProcessingError(f"DICOM to NIfTI conversion failed: {e}") from e
