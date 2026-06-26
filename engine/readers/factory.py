"""Format detection and reader initialization factory."""

from pathlib import Path
from core.exceptions import MRIProcessingError
from engine.readers.base import BaseReader
from engine.readers.dicom import DicomReader
from engine.readers.image import ImageReader
from engine.readers.nifti import NiftiReader


class FormatDetector:
    """Inspects file signatures (magic bytes), headers, and extensions to identify type."""

    @staticmethod
    def detect(path: Path) -> str:
        """Determines scan format: nifti, dicom, or image. Raises MRIProcessingError on invalid formats."""
        if path.is_dir():
            # Check if directory contains any DICOM files
            dicom_files = list(path.glob("*.dcm"))
            if dicom_files:
                return "dicom"
            raise MRIProcessingError(f"Directory {path} does not contain any valid DICOM (.dcm) files.")
        
        if not path.exists():
            raise MRIProcessingError(f"Path does not exist: {path}")

        try:
            with open(path, "rb") as f:
                header = f.read(512)
        except Exception as e:
            raise MRIProcessingError(f"Failed to read file header from {path}: {str(e)}") from e

        # 1. Check PNG signature
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image"
        
        # 2. Check JPG/JPEG signature
        if header.startswith(b"\xff\xd8\xff"):
            return "image"
        
        # 3. Check DICOM magic signature ("DICM" at offset 128)
        if len(header) >= 132 and header[128:132] == b"DICM":
            return "dicom"
        
        # 4. Check NIfTI-1 magic signature ("n+1" or "ni1" at offset 344)
        if len(header) >= 348:
            nifti1_magic = header[344:347]
            if nifti1_magic in (b"n+1", b"ni1"):
                return "nifti"
        
        # 5. Check NIfTI-2 magic signature ("n+2" or "ni2" at offset 4)
        if len(header) >= 8:
            nifti2_magic = header[4:7]
            if nifti2_magic in (b"n+2", b"ni2"):
                return "nifti"

        # 6. Fallback to extension check
        ext = path.name.lower()
        if ext.endswith(".nii") or ext.endswith(".nii.gz"):
            return "nifti"
        if ext.endswith(".dcm"):
            return "dicom"
        if ext.endswith(".png") or ext.endswith(".jpg") or ext.endswith(".jpeg"):
            return "image"

        raise MRIProcessingError(f"Unsupported format or unknown magic bytes for file: {path.name}")


class ReaderFactory:
    """Maps format to reader instances."""

    @staticmethod
    def create(path: Path) -> BaseReader:
        """Instantiates appropriate format reader for input path."""
        fmt = FormatDetector.detect(path)
        if fmt == "nifti":
            return NiftiReader()
        elif fmt == "dicom":
            return DicomReader()
        elif fmt == "image":
            return ImageReader()
        else:
            raise MRIProcessingError(f"Unsupported format: {fmt}")
