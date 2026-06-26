"""Unit tests for the generic dataset compiler framework and ABIDE compiler."""

import json
from pathlib import Path
import pytest
import pandas as pd
from datasets.compiler import compile_dataset, get_compiler
from datasets.abide_compiler import ABIDECompiler
from core.exceptions import ValidationFailedError


@pytest.fixture
def mock_abide_dataset(tmp_path):
    """Creates a temporary mock ABIDE dataset matching BIDS layout and phenotypic CSV."""
    raw_dir = tmp_path / "ABIDE_I"
    raw_dir.mkdir()

    # Generate phenotypic data
    pheno_data = {
        "SUB_ID": [50952, 50953, 50954, 50955, 50956, 50957, 50958, 50959, 50960, 50961],
        "DX_GROUP": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # alternating ASD (1) and CONTROL (2)
        "SEX": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # alternating Male (1) and Female (2)
        "AGE_AT_SCAN": [15.3, 16.4, 12.1, 14.2, 11.5, 22.0, 19.1, 8.5, 10.2, 14.5],
        "SITE_ID": ["NYU", "NYU", "UM_1", "UM_1", "USM", "USM", "NYU", "NYU", "UM_1", "UM_1"],
        "FIQ": [112, 108, 95, 120, -9999, 115, 100, 105, 110, 118],
        "VIQ": [108, 105, 90, 115, 110, 108, 98, 100, 105, 112],
        "PIQ": [115, 110, 100, 122, 105, 120, 102, 110, 112, 120],
    }
    df = pd.DataFrame(pheno_data)
    csv_path = tmp_path / "Phenotypic_V1_0b.csv"
    df.to_csv(csv_path, index=False)

    # Generate BIDS layout directories and dummy T1w files
    for sub_id in pheno_data["SUB_ID"]:
        sub_str = f"sub-00{sub_id}"
        site = "NYU" if sub_id in (50952, 50953, 50958, 50959) else ("UM_1" if sub_id in (50954, 50955, 50960, 50961) else "USM")
        anat_dir = raw_dir / site / sub_str / "anat"
        anat_dir.mkdir(parents=True)
        t1w_file = anat_dir / f"{sub_str}_T1w.nii.gz"
        t1w_file.write_text("dummy NIfTI content")

    return raw_dir, csv_path, tmp_path


def test_abide_compilation_success(mock_abide_dataset):
    """Test successful compilation of ABIDE dataset with correct outputs and rich metadata."""
    raw_dir, csv_path, tmp_path = mock_abide_dataset
    output_dir = tmp_path / "output"

    results = compile_dataset("abide", raw_dir, csv_path, output_dir=output_dir)

    assert results["index_path"].exists()
    assert results["splits_path"].exists()
    assert results["kfold_path"].exists()
    assert results["statistics_path"].exists()

    # Load outputs to verify structure
    with open(results["index_path"]) as f:
        index = json.load(f)
    assert len(index) == 10

    first_entry = index[0]
    assert first_entry["subject_id"] == "sub-0050952"
    assert first_entry["site"] == "NYU"
    assert first_entry["label"] == "ASD"
    assert first_entry["sex"] == "Male"
    assert first_entry["age"] == 15.3
    assert first_entry["fiq"] == 112
    assert first_entry["dataset"] == "ABIDE_I"
    assert first_entry["modality"] == "T1w"

    # Check that -9999 was cleaned to None (which turns to null in JSON)
    fifth_entry = index[4]
    assert fifth_entry["fiq"] is None

    # Load splits
    with open(results["splits_path"]) as f:
        splits = json.load(f)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    # Verify zero subject leakage
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == 10

    # Load K-Fold splits
    with open(results["kfold_path"]) as f:
        kfold = json.load(f)
    assert len(kfold) == 5
    for fold in kfold:
        assert "train" in fold
        assert "val" in fold
        assert len(set(fold["train"]).intersection(set(fold["val"]))) == 0


def test_validation_missing_mri(mock_abide_dataset):
    """Test validation fails if an indexed MRI path does not exist."""
    raw_dir, csv_path, tmp_path = mock_abide_dataset
    output_dir = tmp_path / "output"

    compiler = get_compiler("abide", output_dir=output_dir)
    index = compiler.build_index(raw_dir, csv_path)

    # Delete one MRI file
    Path(index[0]["path"]).unlink()

    with pytest.raises(ValidationFailedError, match="MRI file path does not exist"):
        compiler.validate(index)


def test_validation_mri_without_phenotypic(mock_abide_dataset):
    """Test validation fails if a scanned MRI does not have a phenotypic record."""
    raw_dir, csv_path, tmp_path = mock_abide_dataset
    output_dir = tmp_path / "output"

    # Create an extra BIDS subject folder and T1w scan on disk
    extra_dir = raw_dir / "NYU" / "sub-0050999" / "anat"
    extra_dir.mkdir(parents=True)
    (extra_dir / "sub-0050999_T1w.nii.gz").write_text("extra dummy")

    compiler = get_compiler("abide", output_dir=output_dir)
    with pytest.raises(ValidationFailedError, match="does not have a matching phenotypic CSV record"):
        compiler.build_index(raw_dir, csv_path)


def test_validation_duplicate_detection(mock_abide_dataset):
    """Test duplicate detection raises exception."""
    raw_dir, csv_path, tmp_path = mock_abide_dataset
    compiler = get_compiler("abide", output_dir=tmp_path / "output")
    index = compiler.build_index(raw_dir, csv_path)

    # Inject duplicate subject
    index.append(index[0].copy())
    with pytest.raises(ValidationFailedError, match="Duplicate subject ID found"):
        compiler.validate(index)


def test_malformed_csv_handling(mock_abide_dataset):
    """Test compiler handles malformed CSV headers correctly."""
    raw_dir, csv_path, tmp_path = mock_abide_dataset
    df = pd.read_csv(csv_path)
    # Drop required column
    df = df.drop(columns=["DX_GROUP"])
    bad_csv = tmp_path / "bad_phenotypic.csv"
    df.to_csv(bad_csv, index=False)

    compiler = get_compiler("abide", output_dir=tmp_path / "output")
    with pytest.raises(ValidationFailedError, match="Required column 'DX_GROUP' missing"):
        compiler.build_index(raw_dir, bad_csv)


def test_placeholder_compilers(tmp_path):
    """Verify placeholder compilers raise NotImplementedError."""
    adni = get_compiler("adni", output_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        adni.build_index(tmp_path)

    brats = get_compiler("brats", output_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        brats.build_index(tmp_path)
