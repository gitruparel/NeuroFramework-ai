"""Unit test suite verifying numerical, spatial, and medical preprocessing transforms and inverse playback."""

import logging
from pathlib import Path
import numpy as np
import pytest
import SimpleITK as sitk
import nibabel as nib
from schemas.mri import MRIData, RawMRI
from schemas.processing import ExecutionContext, ProcessingRecord
from schemas.metadata import MRIMetadata, ImageMetadata, PatientMetadata
from schemas.quality import QualityReport
from schemas.validation import ValidationReport, FileValidationReport, MRIValidationReport
from preprocessing.orientation import Reorient
from preprocessing.normalize import IntensityNormalizer
from preprocessing.resample import Resampler
from preprocessing.bias import BiasFieldCorrector
from preprocessing.skull_strip import SkullStripper
from preprocessing.spatial import ForegroundCropper, Pad, CenterCrop, Resize
from preprocessing.pipeline import reverse_preprocessing


@pytest.fixture
def mock_mri_data() -> MRIData:
    """Fixture returning a mock valid MRIData instance (10x12x14 volume)."""
    # Create simple non-RAS affine
    affine = np.array([
        [0.0, 1.5, 0.0, 10.0],
        [-1.5, 0.0, 0.0, 20.0],
        [0.0, 0.0, 2.0, 30.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    raw = RawMRI(
        tensor=np.random.randint(10, 100, size=(10, 12, 14), dtype=np.int16),
        affine=affine,
        header={},
        format="nifti",
        source_path=Path("mock_volume.nii")
    )
    
    metadata = MRIMetadata(
        image=ImageMetadata(voxel_dims=[1.5, 1.5, 2.0], dimensions=[10, 12, 14], modality="T1w"),
        patient=PatientMetadata(patient_id="mock_patient")
    )
    
    quality = QualityReport(
        noise_score=10.0, blur_score=100.0, motion_score=0.1, contrast_score=0.5,
        dynamic_range=90.0, resolution=1.5, slice_count=14, overall_score=0.8
    )
    
    validation = ValidationReport(
        file_validation=FileValidationReport(exists=True, readable=True, header_valid=True, corrupt=False),
        mri_validation=MRIValidationReport(
            voxel_spacing_valid=True, dimensions_valid=True, intensity_valid=True,
            orientation_valid=True, metadata_complete=True, empty_slices_detected=False
        ),
        is_valid=True
    )

    return MRIData(
        raw=raw,
        metadata=metadata,
        quality=quality,
        validation=validation,
        history=[],
        preview=[],
        statistics={"min": 0, "max": 100, "mean": 50, "std": 10, "shape": [10, 12, 14], "dtype": "int16"}
    )


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Fixture returning execution parameters context."""
    logger = logging.getLogger("test_preprocessing")
    return ExecutionContext(
        logger=logger,
        cache=None,
        config={},
        seed=42,
        device="cpu"
    )


def test_reorient_transform(mock_mri_data, execution_context):
    """Verify Reorient transforms grid orientation to RAS and can be inverted."""
    # Ensure active image/affine are initialized
    mock_mri_data.image = mock_mri_data.raw.tensor.copy()
    mock_mri_data.affine = mock_mri_data.raw.affine.copy()
    
    orig_ornt = nib.orientations.io_orientation(mock_mri_data.affine)
    orig_axcodes = nib.orientations.ornt2axcodes(orig_ornt)
    assert orig_axcodes != ('R', 'A', 'S')
    
    transform = Reorient(target="RAS")
    res = transform.process(mock_mri_data, execution_context)
    
    # Check that output is RAS oriented
    new_ornt = nib.orientations.io_orientation(res.affine)
    new_axcodes = nib.orientations.ornt2axcodes(new_ornt)
    assert new_axcodes == ('R', 'A', 'S')
    
    # Check that raw data remained immutable
    assert np.array_equal(res.raw.tensor, mock_mri_data.raw.tensor)
    assert np.array_equal(res.raw.affine, mock_mri_data.raw.affine)
    
    # Perform reverse playback
    reversed_data = reverse_preprocessing(res)
    rev_ornt = nib.orientations.io_orientation(reversed_data.affine)
    rev_axcodes = nib.orientations.ornt2axcodes(rev_ornt)
    assert rev_axcodes == orig_axcodes
    assert np.array_equal(reversed_data.image, mock_mri_data.image)


def test_intensity_normalization(mock_mri_data, execution_context):
    """Verify IntensityNormalizer correctly rescales and preserves raw immutability."""
    mock_mri_data.image = mock_mri_data.raw.tensor.copy()
    mock_mri_data.affine = mock_mri_data.raw.affine.copy()
    
    # Z-Score
    norm_z = IntensityNormalizer(mode="z_score")
    res_z = norm_z.process(mock_mri_data, execution_context)
    assert pytest.approx(float(np.mean(res_z.image)), abs=1e-5) == 0.0
    assert pytest.approx(float(np.std(res_z.image)), abs=1e-5) == 1.0
    
    # Min-Max
    norm_mm = IntensityNormalizer(mode="min_max")
    res_mm = norm_mm.process(mock_mri_data, execution_context)
    assert float(np.min(res_mm.image)) >= 0.0
    assert float(np.max(res_mm.image)) <= 1.0
    
    # Check raw tensor is unchanged
    assert np.array_equal(res_z.raw.tensor, mock_mri_data.raw.tensor)


def test_resampler_transform(mock_mri_data, execution_context):
    """Verify Resampler shifts grid spacing and maps to/from SimpleITK correctly."""
    # Use a constant volume to make linear interpolation exact and reversible
    mock_mri_data.image = np.ones((10, 12, 14), dtype=np.float32) * 50.0
    mock_mri_data.affine = mock_mri_data.raw.affine.copy()
    
    resample = Resampler(spacing=[1.0, 1.0, 1.0])
    res = resample.process(mock_mri_data, execution_context)
    
    assert res.metadata.image.voxel_dims == [1.0, 1.0, 1.0]
    
    # Test inverse spacing/shape restoration
    reversed_data = reverse_preprocessing(res)
    assert reversed_data.metadata.image.voxel_dims == [1.5, 1.5, 2.0]
    assert reversed_data.image.shape == mock_mri_data.image.shape
    assert np.allclose(reversed_data.image, mock_mri_data.image, atol=1e-2)


def test_spatial_cropper_and_pad(mock_mri_data, execution_context):
    """Verify cropping and padding change shape and coordinate mappings correctly."""
    # Create mock scan with localized foreground at center
    tensor = np.zeros((12, 14, 16), dtype=np.int16)
    tensor[3:9, 4:10, 5:12] = 100
    mock_mri_data.image = tensor
    mock_mri_data.affine = mock_mri_data.raw.affine.copy()
    mock_mri_data.metadata.image.dimensions = [12, 14, 16]
    
    # 1. Bounding-Box Foreground Cropping
    cropper = ForegroundCropper()
    cropped = cropper.process(mock_mri_data, execution_context)
    assert cropped.image.shape == (6, 6, 7)
    
    # Verify inverse uncrops perfectly
    rev_cropped = reverse_preprocessing(cropped)
    assert rev_cropped.image.shape == (12, 14, 16)
    assert np.array_equal(rev_cropped.image, mock_mri_data.image)
    assert np.array_equal(rev_cropped.affine, mock_mri_data.affine)
    
    # 2. Symmetrical Padding
    padder = Pad(target_shape=[20, 20, 20])
    padded = padder.process(mock_mri_data, execution_context)
    assert padded.image.shape == (20, 20, 20)
    
    # Verify inverse pads back
    rev_padded = reverse_preprocessing(padded)
    assert rev_padded.image.shape == (12, 14, 16)
    assert np.array_equal(rev_padded.image, mock_mri_data.image)
    assert np.array_equal(rev_padded.affine, mock_mri_data.affine)
    
    # 3. Center Cropping
    cc = CenterCrop(target_shape=[6, 6, 6])
    center_cropped = cc.process(mock_mri_data, execution_context)
    assert center_cropped.image.shape == (6, 6, 6)
    
    rev_cc = reverse_preprocessing(center_cropped)
    assert rev_cc.image.shape == (12, 14, 16)
    # The cropped spatial region is preserved and restored
    assert np.array_equal(rev_cc.image[3:9, 4:10, 5:11], mock_mri_data.image[3:9, 4:10, 5:11])
    # The discarded areas must be zero-padded in the restored space
    mask = np.ones((12, 14, 16), dtype=bool)
    mask[3:9, 4:10, 5:11] = False
    assert np.all(rev_cc.image[mask] == 0)


def test_bias_field_corrector(mock_mri_data, execution_context):
    """Verify N4 Bias corrector runs and leverages brain mask filter."""
    mock_mri_data.image = mock_mri_data.raw.tensor.copy()
    mock_mri_data.affine = mock_mri_data.raw.affine.copy()
    
    corrector = BiasFieldCorrector(strategy="n4")
    
    # Test without mask
    res_no_mask = corrector.process(mock_mri_data, execution_context)
    assert res_no_mask.image.shape == mock_mri_data.image.shape
    
    # Test with mask
    mock_mri_data.brain_mask = np.ones_like(mock_mri_data.image)
    res_mask = corrector.process(mock_mri_data, execution_context)
    assert res_mask.image.shape == mock_mri_data.image.shape


def test_skull_stripper_strategies(mock_mri_data, execution_context):
    """Verify SkullStripper supports Otsu baseline and logs placeholders warnings."""
    # Create simple binary region image
    tensor = np.zeros((10, 10, 10), dtype=np.int16)
    tensor[3:7, 3:7, 3:7] = 500
    mock_mri_data.image = tensor
    mock_mri_data.affine = mock_mri_data.raw.affine.copy()
    
    stripper = SkullStripper(strategy="threshold")
    res = stripper.process(mock_mri_data, execution_context)
    
    assert res.brain_mask is not None
    assert np.array_equal(res.image, mock_mri_data.image * res.brain_mask)
    assert set(np.unique(res.brain_mask)).issubset({0.0, 1.0})
    
    # Verify DL placeholders fall back to Otsu gracefully
    stripper_synth = SkullStripper(strategy="synthstrip")
    res_synth = stripper_synth.process(mock_mri_data, execution_context)
    assert res_synth.brain_mask is not None
    
    stripper_fast = SkullStripper(strategy="fastsurfer")
    res_fast = stripper_fast.process(mock_mri_data, execution_context)
    assert res_fast.brain_mask is not None
    
    
def test_resize_transform(mock_mri_data, execution_context):
    """Verify Resize shifts grid size and maps to/from SimpleITK correctly and is reversible."""
    mock_mri_data.image = np.ones((10, 12, 14), dtype=np.float32) * 50.0
    mock_mri_data.affine = mock_mri_data.raw.affine.copy()
    mock_mri_data.metadata.image.dimensions = [10, 12, 14]
    
    resize = Resize(target_shape=[16, 18, 20])
    res = resize.process(mock_mri_data, execution_context)
    
    assert res.image.shape == (16, 18, 20)
    
    # Test inverse shape restoration
    reversed_data = reverse_preprocessing(res)
    assert reversed_data.image.shape == (10, 12, 14)
    assert np.allclose(reversed_data.image, mock_mri_data.image, atol=1e-2)
