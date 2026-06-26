"""Reader implementation for JPG/PNG standard image formats."""

from pathlib import Path
import cv2
import numpy as np
from core.exceptions import MRIProcessingError
from engine.readers.base import BaseReader
from schemas.mri import RawMRI


class ImageReader(BaseReader):
    """Loads 2D standard images (JPG, PNG) using OpenCV."""

    def read(self, path: Path) -> RawMRI:
        try:
            # Load image as grayscale
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise MRIProcessingError(f"OpenCV failed to read image at {path}")
            
            # Convert 2D image (y, x) to tensor layout (x, y, 1) to match 3D volume style
            tensor = np.transpose(image, (1, 0))
            tensor = np.expand_dims(tensor, axis=-1)
            
            # Image files have identity spatial layout mapping
            affine = np.eye(4)
            
            header = {
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": 1,
            }
            
            return RawMRI(
                tensor=tensor,
                affine=affine,
                header=header,
                format="image",
                source_path=path,
            )
        except Exception as e:
            raise MRIProcessingError(f"ImageReader failed loading from {path}: {str(e)}") from e
