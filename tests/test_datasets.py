"""Unit and integration testing suite for Stage 2 Dataset Management Engine."""

import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from core.exceptions import ConfigurationError, MRIProcessingError
from datasets.base import DatasetRegistry
from datasets.manager import DatasetManager, SplitManager
from schemas.dataset import DatasetSample


@pytest.fixture
def mock_datasets_dir(tmp_path) -> Path:
    """Fixture creating mock folders for ABIDE, ADNI, and BraTS datasets."""
    root = tmp_path / "mock_raw_data"
    root.mkdir()

    # 1. ABIDE Autism Mock Dataset
    abide_dir = root / "abide"
    abide_dir.mkdir()
    
    # Valid CONTROL subject
    sub_c1 = abide_dir / "CONTROL" / "sub-ab01"
    sub_c1.mkdir(parents=True)
    nii_img = nib.Nifti1Image(np.random.randint(0, 100, size=(5, 5, 5), dtype=np.int16), affine=np.eye(4))
    nib.save(nii_img, str(sub_c1 / "scan.nii.gz"))

    # Valid ASD subject
    sub_a1 = abide_dir / "ASD" / "sub-ab02"
    sub_a1.mkdir(parents=True)
    nib.save(nii_img, str(sub_a1 / "scan.nii.gz"))

    # Empty subject folder edge-case (should be ignored by index builder)
    sub_empty = abide_dir / "CONTROL" / "sub-empty"
    sub_empty.mkdir(parents=True)

    # 2. ADNI Alzheimer Mock Dataset (mixed NIfTI + DICOM)
    adni_dir = root / "adni"
    adni_dir.mkdir()

    # CN NIfTI subject
    sub_cn = adni_dir / "CN" / "sub-adcn"
    sub_cn.mkdir(parents=True)
    nib.save(nii_img, str(sub_cn / "scan.nii"))

    # AD DICOM subject (using uint16 for SimpleITK writing compatibility)
    sub_ad = adni_dir / "AD" / "sub-adad"
    sub_ad.mkdir(parents=True)
    sitk_img = sitk.GetImageFromArray(np.random.randint(0, 100, size=(5, 5, 5), dtype=np.uint16))
    sitk.WriteImage(sitk_img, str(sub_ad / "slice.dcm"))

    # 3. BraTS Tumor Mock Dataset (Multimodal & segmentation targets)
    brats_dir = root / "brats"
    brats_dir.mkdir()

    # Valid BraTS patient (all 5 modalities)
    sub_br1 = brats_dir / "BraTS_001"
    sub_br1.mkdir(parents=True)
    nib.save(nii_img, str(sub_br1 / "BraTS_001_t1.nii.gz"))
    nib.save(nii_img, str(sub_br1 / "BraTS_001_t1ce.nii.gz"))
    nib.save(nii_img, str(sub_br1 / "BraTS_001_t2.nii.gz"))
    nib.save(nii_img, str(sub_br1 / "BraTS_001_flair.nii.gz"))
    nib.save(nii_img, str(sub_br1 / "BraTS_001_seg.nii.gz"))

    # Invalid BraTS patient (missing FLAIR modality channel)
    sub_br2 = brats_dir / "BraTS_002"
    sub_br2.mkdir(parents=True)
    nib.save(nii_img, str(sub_br2 / "BraTS_002_t1.nii.gz"))
    nib.save(nii_img, str(sub_br2 / "BraTS_002_t1ce.nii.gz"))
    nib.save(nii_img, str(sub_br2 / "BraTS_002_t2.nii.gz"))
    nib.save(nii_img, str(sub_br2 / "BraTS_002_seg.nii.gz")) # missing flair!

    return root


def test_split_manager_reproducibility():
    """Verify SplitManager split distributions are reproducible with fixed seeds."""
    manager = SplitManager()
    subjects = [f"sub-{i:03d}" for i in range(100)]

    # Splitting twice with same seed must return identical dict contents
    split_a1 = manager.split_subjects(subjects, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=123)
    split_a2 = manager.split_subjects(subjects, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=123)
    assert split_a1 == split_a2

    # Splitting with different seed must return different contents
    split_b = manager.split_subjects(subjects, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=999)
    assert split_a1 != split_b


def test_split_manager_invalid_ratios():
    """Verify SplitManager validates split bounds and ratio sums."""
    manager = SplitManager()
    subjects = ["sub-01", "sub-02", "sub-03"]

    # Ratios that do not sum to 1.0
    with pytest.raises(ConfigurationError):
        manager.split_subjects(subjects, train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)

    # Ratios outside bounds
    with pytest.raises(ConfigurationError):
        manager.split_subjects(subjects, train_ratio=-0.2, val_ratio=0.6, test_ratio=0.6)


def test_dataset_manager_orchestration(mock_datasets_dir, tmp_path):
    """Verify DatasetManager correctly indexes directories and serializes index, stats, and splits."""
    manager = DatasetManager(data_root=tmp_path)
    
    # Process ABIDE
    res = manager.process_dataset("abide", mock_datasets_dir / "abide")
    
    assert Path(res["index_path"]).exists()
    assert Path(res["statistics_path"]).exists()
    assert Path(res["splits_path"]).exists()
    
    # Statistics: verify empty folder is ignored
    stats = res["statistics"]
    assert stats["total_files"] == 2  # CONTROL sub-ab01, ASD sub-ab02 (sub-empty ignored)
    assert stats["unique_subjects"] == 2
    assert stats["class_distribution"] == {"CONTROL": 1, "ASD": 1}


def test_abide_pytorch_loader(mock_datasets_dir, tmp_path):
    """Verify ABIDEDataset resolves configuration label maps and returns standard DatasetSample model."""
    manager = DatasetManager(data_root=tmp_path)
    res = manager.process_dataset("abide", mock_datasets_dir / "abide", train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)

    # Instantiate via Registry
    label_map = {"CONTROL": 0, "ASD": 1}
    dataset = DatasetRegistry.get(
        "abide",
        index_file=res["index_path"],
        split_file=res["splits_path"],
        split_name="train",
        label_map=label_map
    )

    assert len(dataset) == 1
    sample = dataset[0]
    
    assert isinstance(sample, DatasetSample)
    assert sample.image.shape == (1, 5, 5, 5)  # Singleton channel added
    assert sample.label in (0, 1)
    assert sample.dataset_name == "abide"
    assert sample.mask is None


def test_adni_mixed_formats_loader(mock_datasets_dir, tmp_path):
    """Verify ADNIDataset loads both NIfTI and DICOM volumes correctly."""
    manager = DatasetManager(data_root=tmp_path)
    res = manager.process_dataset("adni", mock_datasets_dir / "adni")

    label_map = {"CN": 0, "AD": 1}
    dataset = DatasetRegistry.get(
        "adni",
        index_file=res["index_path"],
        label_map=label_map
    )

    # 2 indexed items: NIfTI scan and DICOM slice
    assert len(dataset) == 2
    
    # Sort order is sub-adad first (AD), sub-adcn second (CN)
    sample_ad = dataset[0]
    assert sample_ad.label == 1
    assert sample_ad.image.shape == (1, 5, 5, 5)

    sample_cn = dataset[1]
    assert sample_cn.label == 0
    assert sample_cn.image.shape == (1, 5, 5, 5)


def test_brats_multimodal_loader(mock_datasets_dir, tmp_path):
    """Verify BraTSDataset stacks 4 modalities and flags missing channels."""
    manager = DatasetManager(data_root=tmp_path)
    res = manager.process_dataset("brats", mock_datasets_dir / "brats")

    dataset = DatasetRegistry.get(
        "brats",
        index_file=res["index_path"]
    )

    # Both patient directories indexed
    assert len(dataset) == 2

    # BraTS_001 has all modalities and target mask
    sample_valid = dataset[0]
    assert sample_valid.image.shape == (4, 5, 5, 5)  # Stacked channels
    assert sample_valid.mask is not None
    assert sample_valid.mask.shape == (1, 5, 5, 5)  # segmentation mask

    # BraTS_002 is missing FLAIR channel. Attempting to load it must raise MRIProcessingError
    with pytest.raises(MRIProcessingError) as exc_info:
        _ = dataset[1]
    assert "Missing required modality channel 'flair'" in str(exc_info.value)
