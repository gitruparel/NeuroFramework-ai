"""Unit tests for the Dataset Audit Engine."""

from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from engine.audit import DatasetAuditor


@pytest.fixture
def mock_dataset_root(tmp_path) -> Path:
    """Creates a temporary mock medical dataset containing AD and CN subjects."""
    dataset_dir = tmp_path / "ADNI_Mock"
    dataset_dir.mkdir()

    # 1. Create a valid Alzheimer (AD) subject NIfTI scan
    sub_1_dir = dataset_dir / "AD" / "sub-01"
    sub_1_dir.mkdir(parents=True)
    nifti_data = np.random.randint(0, 100, size=(8, 8, 8), dtype=np.int16)
    nifti_img = nib.Nifti1Image(nifti_data, affine=np.eye(4))
    nib.save(nifti_img, str(sub_1_dir / "t1.nii.gz"))

    # 2. Create a corrupted file inside AD subject folder
    sub_2_dir = dataset_dir / "AD" / "sub-02"
    sub_2_dir.mkdir(parents=True)
    with open(sub_2_dir / "broken_scan.nii.gz", "w") as f:
        f.write("Corrupt non-binary text content simulation")

    # 3. Create a valid Control Normal (CN) subject NIfTI scan
    sub_3_dir = dataset_dir / "CN" / "sub-03"
    sub_3_dir.mkdir(parents=True)
    nib.save(nifti_img, str(sub_3_dir / "t1.nii.gz"))

    return dataset_dir


def test_dataset_auditor(mock_dataset_root):
    """Verify DatasetAuditor correctly audits formats, subjects, corruptions, and class balance."""
    auditor = DatasetAuditor()
    report = auditor.audit(mock_dataset_root)

    # 3 unique subjects folders (sub-01, sub-02, sub-03)
    assert report.total_subjects == 3
    assert report.total_files == 3

    # formats should include nifti
    assert "nifti" in report.formats_present

    # corrupt scan count
    assert len(report.corrupt_files) == 1
    assert "broken_scan.nii.gz" in report.corrupt_files[0]

    # class balance: should infer 1 AD (sub-01) and 1 CN (sub-03) correctly
    assert report.class_balance.get("AD") == 1
    assert report.class_balance.get("CN") == 1

    # spacing statistics should exist
    assert report.voxel_spacings is not None
    assert report.voxel_spacings.mean_spacing == [1.0, 1.0, 1.0]
