"""Medical image loading interfaces (NIfTI, DICOM)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import nibabel as nib
import SimpleITK as sitk
from core.exceptions import MRIProcessingError


class BaseMRIReader(ABC):
    """Abstract interface for MRI loaders."""

    @abstractmethod
    def read(self, path: Path | str) -> Any:
        """Reads scan file returning structural imaging object."""
        pass


class NibabelReader(BaseMRIReader):
    """Reads NIfTI volumes using NiBabel."""

    def read(self, path: Path | str) -> nib.spatialimages.SpatialImage:
        try:
            return nib.load(str(path))
        except Exception as e:
            raise MRIProcessingError(f"NiBabel failed loading MRI at {path}: {e}") from e


class SimpleITKReader(BaseMRIReader):
    """Reads DICOM/NIfTI scans using SimpleITK."""

    def read(self, path: Path | str) -> sitk.Image:
        try:
            return sitk.ReadImage(str(path))
        except Exception as e:
            raise MRIProcessingError(f"SimpleITK failed loading MRI at {path}: {e}") from e
