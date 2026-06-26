"""Unit tests for the Universal MRI Processing Engine ingestion pipeline."""

import os
from pathlib import Path
import cv2
import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from core.exceptions import MRIProcessingError
from engine.pipeline import MRIEngine
from engine.readers.factory import FormatDetector, ReaderFactory


@pytest.fixture
def mock_files_dir(tmp_path) -> Path:
    """Fixture creating directory containing valid mock medical and 2D images."""
    # Create temp directory layout
    test_dir = tmp_path / "mock_scans"
    test_dir.mkdir()

    # 1. Create a mock 3D NIfTI scan (10x10x10)
    nifti_data = np.random.randint(0, 1000, size=(10, 10, 10), dtype=np.int16)
    nifti_img = nib.Nifti1Image(nifti_data, affine=np.eye(4))
    nib.save(nifti_img, str(test_dir / "brain.nii.gz"))
    nib.save(nifti_img, str(test_dir / "brain.nii"))

    # 2. Create a mock 3D DICOM scan (10x10x10) using SimpleITK
    sitk_img = sitk.GetImageFromArray(np.random.randint(0, 1000, size=(10, 10, 10), dtype=np.uint16))
    sitk_img.SetSpacing([1.2, 1.2, 1.2])
    sitk_img.SetOrigin([1.0, 2.0, 3.0])
    sitk.WriteImage(sitk_img, str(test_dir / "slice.dcm"))

    # 3. Create mock standard 2D images (JPG & PNG)
    img_data = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "photo.jpg"), img_data)
    cv2.imwrite(str(test_dir / "photo.png"), img_data)

    return test_dir


def test_format_detector_magic_bytes(mock_files_dir):
    """Verify FormatDetector prioritizes magic bytes over extension."""
    nifti_path = mock_files_dir / "brain.nii"
    jpg_path = mock_files_dir / "photo.jpg"

    # Rename NIfTI scan to JPG extension
    renamed_nifti = mock_files_dir / "lying_brain.jpg"
    os.rename(nifti_path, renamed_nifti)

    # Rename JPG photo to NIfTI extension
    renamed_jpg = mock_files_dir / "lying_photo.nii.gz"
    os.rename(jpg_path, renamed_jpg)

    # Verify detection matches real magic content, not the extension
    assert FormatDetector.detect(renamed_nifti) == "nifti"
    assert FormatDetector.detect(renamed_jpg) == "image"


def test_nifti_engine_loading(mock_files_dir):
    """Verify MRIEngine successfully loads and processes NIfTI files into MRIData."""
    engine = MRIEngine()
    data = engine.load(mock_files_dir / "brain.nii.gz")

    assert data.raw.format == "nifti"
    assert data.statistics.shape == [10, 10, 10]
    assert data.statistics.min >= 0.0
    assert len(data.preview) == 3  # Axial, coronal, sagittal orthogonal previews
    assert data.validation.is_valid is True


def test_dicom_engine_loading(mock_files_dir):
    """Verify MRIEngine successfully loads and processes DICOM files into MRIData."""
    engine = MRIEngine()
    data = engine.load(mock_files_dir / "slice.dcm")

    assert data.raw.format == "dicom"
    assert data.statistics.shape == [10, 10, 10]
    assert list(data.metadata.image.voxel_dims) == pytest.approx([1.2, 1.2, 1.2])
    assert data.validation.is_valid is True


def test_image_engine_loading(mock_files_dir):
    """Verify MRIEngine successfully loads and processes standard 2D images."""
    engine = MRIEngine()
    data = engine.load(mock_files_dir / "photo.png")

    assert data.raw.format == "image"
    assert data.statistics.shape == [50, 50, 1]
    assert len(data.preview) == 1  # 2D preview only
    assert data.validation.is_valid is True


def test_nonexistent_file():
    """Verify loading a nonexistent file raises appropriate exception or fails validation."""
    engine = MRIEngine()
    with pytest.raises(MRIProcessingError):
        engine.load(Path("nonexistent_scan.nii.gz"))


def test_caching_mechanism(mock_files_dir):
    """Verify MRIEngine caches outputs and returns identical references on reload."""
    engine = MRIEngine()
    path = mock_files_dir / "photo.png"

    data1 = engine.load(path)
    data2 = engine.load(path)

    # Objects should share identical identity reference
    assert data1 is data2
