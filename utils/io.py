"""File IO load/save utilities for medical files."""

from pathlib import Path
from typing import Any
import nibabel as nib


def save_nifti(data: Any, header: Any, affine: Any, output_path: Path | str) -> None:
    """Helper to export numpy arrays back into NIfTI volumes."""
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(output_path))
