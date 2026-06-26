"""Reader implementation for DICOM file formats."""

from pathlib import Path
import SimpleITK as sitk
import numpy as np
from core.exceptions import MRIProcessingError
from engine.readers.base import BaseReader
from schemas.mri import RawMRI


class DicomReader(BaseReader):
    """Loads DICOM volumes using SimpleITK."""

    def read(self, path: Path) -> RawMRI:
        try:
            reader = sitk.ImageSeriesReader()
            
            if path.is_dir():
                dicom_names = reader.GetGDCMSeriesFileNames(str(path))
                if not dicom_names:
                    raise MRIProcessingError(f"No DICOM series files detected inside directory {path}")
                reader.SetFileNames(dicom_names)
                sitk_image = reader.Execute()
            else:
                sitk_image = sitk.ReadImage(str(path))
            
            # SimpleITK image to NumPy array is typically transposed (z, y, x)
            # Transpose to match standard NIfTI coordinates shape (x, y, z) if 3D
            tensor = sitk.GetArrayFromImage(sitk_image)
            if len(tensor.shape) == 3:
                tensor = np.transpose(tensor, (2, 1, 0))
            
            # Construct metadata dictionary from DICOM keys
            header = {}
            for key in sitk_image.GetMetaDataKeys():
                header[key] = sitk_image.GetMetaData(key)

            # Build dummy affine coordinate matrix from direction and spacing
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = sitk_image.GetDirection()
            
            affine = np.eye(4)
            if len(spacing) == 3:
                dir_matrix = np.array(direction).reshape(3, 3)
                affine[:3, :3] = dir_matrix * np.array(spacing)
                affine[:3, 3] = origin

            return RawMRI(
                tensor=tensor,
                affine=affine,
                header=header,
                format="dicom",
                source_path=path,
            )
        except Exception as e:
            raise MRIProcessingError(f"DicomReader failed loading from {path}: {str(e)}") from e
