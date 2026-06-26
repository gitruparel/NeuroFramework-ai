"""Reader implementation for NIfTI file formats."""

from pathlib import Path
import nibabel as nib
import numpy as np
from core.exceptions import MRIProcessingError
from engine.readers.base import BaseReader
from schemas.mri import RawMRI


class NiftiReader(BaseReader):
    """Loads 3D structural volumes from NIfTI files (.nii, .nii.gz)."""

    def read(self, path: Path) -> RawMRI:
        try:
            try:
                nifti_img = nib.load(str(path))
                tensor = np.asanyarray(nifti_img.dataobj)
                affine = nifti_img.affine
            except nib.filebasedimages.ImageFileError:
                # Handle non-standard file extensions containing NIfTI data
                with open(path, "rb") as f:
                    signature = f.read(2)
                is_gzipped = (signature == b"\x1f\x8b")
                
                if is_gzipped:
                    import gzip
                    with gzip.open(path, "rb") as f:
                        header = f.read(348)
                else:
                    with open(path, "rb") as f:
                        header = f.read(348)
                
                if len(header) >= 8 and header[4:7] in (b"n+2", b"ni2"):
                    img_klass = nib.Nifti2Image
                else:
                    img_klass = nib.Nifti1Image
                
                if is_gzipped:
                    import gzip
                    f_gz = gzip.open(path, "rb")
                    try:
                        fh = nib.FileHolder(fileobj=f_gz)
                        nifti_img = img_klass.from_file_map({'image': fh})
                        tensor = np.asanyarray(nifti_img.dataobj)
                        affine = nifti_img.affine
                    finally:
                        f_gz.close()
                else:
                    fh = nib.FileHolder(filename=str(path))
                    nifti_img = img_klass.from_file_map({'image': fh})
                    tensor = np.asanyarray(nifti_img.dataobj)
                    affine = nifti_img.affine
            
            # Convert header fields safely to dictionary
            header_dict = {}
            if hasattr(nifti_img, "header"):
                for key in nifti_img.header.keys():
                    try:
                        header_dict[key] = nifti_img.header[key].tolist()
                    except AttributeError:
                        header_dict[key] = nifti_img.header[key]
                    except Exception:
                        pass
            
            return RawMRI(
                tensor=tensor,
                affine=affine,
                header=header_dict,
                format="nifti",
                source_path=path,
            )
        except Exception as e:
            raise MRIProcessingError(f"NiftiReader failed loading NIfTI from {path}: {str(e)}") from e
