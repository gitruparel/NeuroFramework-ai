"""Manual stress testing script for Stage 1 (UMPE) and Stage 1.5 (Dataset Auditor).

Generates various mock clinical scenarios programmatically and prints ingestion metrics.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2

# Add root folder to PYTHONPATH to allow core/engine imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.pipeline import MRIEngine
from engine.audit import DatasetAuditor
from core.exceptions import BrainAIError


def generate_test_dataset(root_path: Path):
    """Generates mock dataset subjects spanning all required stress cases."""
    print("Generating programmatic mock datasets...")

    # 1. Valid Alzheimer's (AD) subject with NIfTI.gz
    ad_dir = root_path / "AD" / "sub-ad01"
    ad_dir.mkdir(parents=True, exist_ok=True)
    nii_data = np.random.randint(10, 500, size=(12, 12, 12), dtype=np.int16)
    nii_img = nib.Nifti1Image(nii_data, affine=np.eye(4))
    nib.save(nii_img, str(ad_dir / "t1_scan.nii.gz"))

    # 2. Valid Control Normal (CN) subject with uncompressed NIfTI
    cn_dir = root_path / "CN" / "sub-cn02"
    cn_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nii_img, str(cn_dir / "t1_scan.nii"))

    # 3. Valid DICOM subject (using uint16 for Windows ITK compatibility)
    dicom_dir = root_path / "CN" / "sub-cn03"
    dicom_dir.mkdir(parents=True, exist_ok=True)
    sitk_img = sitk.GetImageFromArray(np.random.randint(10, 500, size=(10, 10, 10), dtype=np.uint16))
    sitk_img.SetSpacing([1.2, 1.2, 1.5])
    # Add scanner tags to simulate metadata extraction
    sitk_img.SetMetaData("0008|0070", "Siemens")
    sitk_img.SetMetaData("0008|1090", "Prisma_fit")
    sitk_img.SetMetaData("0010|0020", "PATIENT-CN03")
    sitk.WriteImage(sitk_img, str(dicom_dir / "slice.dcm"))

    # 4. Valid standard 2D image formats
    image_dir = root_path / "Normal" / "sub-img04"
    image_dir.mkdir(parents=True, exist_ok=True)
    img_data = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
    cv2.imwrite(str(image_dir / "axial_slice.png"), img_data)
    cv2.imwrite(str(image_dir / "axial_slice.jpg"), img_data)

    # 5. Faulty Case: Corrupted file signature
    corrupt_dir = root_path / "AD" / "sub-broken05"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    with open(corrupt_dir / "corrupted_scan.nii.gz", "w") as f:
        f.write("This is not a binary NIfTI volume, it is text!")

    # 6. Faulty Case: Constant flat intensity array (fails validation checks)
    flat_dir = root_path / "CN" / "sub-flat06"
    flat_dir.mkdir(parents=True, exist_ok=True)
    flat_data = np.ones((8, 8, 8), dtype=np.int16) * 100
    flat_img = nib.Nifti1Image(flat_data, affine=np.eye(4))
    nib.save(flat_img, str(flat_dir / "flat_scan.nii"))

    # 7. Faulty Case: Blank zero-intensity slices inside the scan volume
    blank_dir = root_path / "AD" / "sub-blank07"
    blank_dir.mkdir(parents=True, exist_ok=True)
    blank_data = np.random.randint(10, 500, size=(8, 8, 8), dtype=np.int16)
    blank_data[:, :, 4] = 0  # Make slice index 4 completely empty/blank
    blank_img = nib.Nifti1Image(blank_data, affine=np.eye(4))
    nib.save(blank_img, str(blank_dir / "blank_slice.nii"))

    # 8. Faulty Case: Wrong extension naming (photo.jpg containing uncompressed NIfTI)
    lying_dir = root_path / "Normal" / "sub-lying08"
    lying_dir.mkdir(parents=True, exist_ok=True)
    temp_nii = lying_dir / "temp.nii"
    nib.save(nii_img, str(temp_nii))
    temp_nii.rename(lying_dir / "lying_brain.jpg")

    print("Generation complete.\n")


def execute_manual_tests(root_path: Path):
    """Executes the validation checklist against generated cases."""
    engine = MRIEngine()
    auditor = DatasetAuditor()

    print("=" * 60)
    print("RUNNING PIPELINE LOADER STRESS TESTS")
    print("=" * 60)

    test_paths = [
        ("NIfTI.gz [Valid AD]", root_path / "AD" / "sub-ad01" / "t1_scan.nii.gz"),
        ("NIfTI [Valid CN]", root_path / "CN" / "sub-cn02" / "t1_scan.nii"),
        ("DICOM [Valid CN]", root_path / "CN" / "sub-cn03" / "slice.dcm"),
        ("PNG [Valid 2D]", root_path / "Normal" / "sub-img04" / "axial_slice.png"),
        ("JPG [Valid 2D]", root_path / "Normal" / "sub-img04" / "axial_slice.jpg"),
        ("Flat Array [Fails Validation]", root_path / "CN" / "sub-flat06" / "flat_scan.nii"),
        ("Blank Slices [Fails Validation]", root_path / "AD" / "sub-blank07" / "blank_slice.nii"),
        ("Renamed Format [Prioritizes Header]", root_path / "Normal" / "sub-lying08" / "lying_brain.jpg"),
    ]

    for label, path in test_paths:
        print(f"\n---> Testing target: {label}")
        try:
            mri_data = engine.load(path)
            print(f"  Format Detected: {mri_data.raw.format}")
            print(f"  Shape: {mri_data.statistics.shape}")
            print(f"  Min/Max/Mean: {mri_data.statistics.min:.1f} / {mri_data.statistics.max:.1f} / {mri_data.statistics.mean:.1f}")
            print(f"  Valid: {mri_data.validation.is_valid}")
            print(f"  Validation Errors: {mri_data.validation.errors}")
            print(f"  Overall Quality Score: {mri_data.quality.overall_score:.4f}")
            print(f"  Pipeline History: {mri_data.history}")
        except Exception as e:
            print(f"  [CRITICAL LOAD FAILURE] {type(e).__name__}: {e}")

    # Test explicit corrupted load handling
    print("\n---> Testing target: Corrupted binary file (should fail reading)")
    try:
        engine.load(root_path / "AD" / "sub-broken05" / "corrupted_scan.nii.gz")
    except BrainAIError as e:
        print(f"  Passed (raised correct platform exception): {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  FAILED (raised incorrect exception): {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("RUNNING DATASET AUDITOR")
    print("=" * 60)

    try:
        report = auditor.audit(root_path)
        print(f"Total files audited: {report.total_files}")
        print(f"Total subjects grouped: {report.total_subjects}")
        print(f"Formats present: {report.formats_present}")
        print(f"Class Balance: {report.class_balance}")
        print(f"Corrupt Files Count: {len(report.corrupt_files)}")
        print(f"Scanner Manufacturers: {report.scanner_manufacturers}")
        if report.voxel_spacings:
            print(f"Voxel Spacing (Mean): {report.voxel_spacings.mean_spacing}")
        print(f"Missing Metadata Count: {report.missing_metadata_count}")
    except Exception as e:
        print(f"  [AUDITOR FAILURE] {type(e).__name__}: {e}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generate_test_dataset(temp_path)
        execute_manual_tests(temp_path)
