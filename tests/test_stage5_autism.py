"""Integration smoke test validating Stage 5 Autism model preprocessing, training, reporting, and resumption."""

import json
from pathlib import Path
import pytest
import torch
import nibabel as nib
import numpy as np
import yaml

from training.train_autism import run_training_experiment


def test_autism_training_smoke_test(tmp_path):
    """Verify that train_autism.py runs end-to-end with offline preprocessing and training resumption."""
    # 1. Create directories
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    experiment_dir = tmp_path / "experiment"
    
    # 2. Generate 20 mock subjects (10 CONTROL, 10 ASD)
    subjects = []
    index_data = []
    
    # Set seed for reproducible numpy generation
    np.random.seed(42)
    
    for i in range(1, 21):
        sub_id = f"sub-ab{i:02d}"
        label = "CONTROL" if i <= 10 else "ASD"
        sub_dir = raw_dir / label / sub_id
        sub_dir.mkdir(parents=True)
        
        # Write small random 3D NIfTI file (shape 32, 32, 32)
        img_path = sub_dir / "scan.nii.gz"
        data = np.random.randint(0, 100, size=(32, 32, 32), dtype=np.int16)
        nii_img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(nii_img, str(img_path))
        
        index_data.append({
            "subject_id": sub_id,
            "path": str(img_path),
            "label": label
        })
        subjects.append(sub_id)
        
    index_file = tmp_path / "abide_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=4)
        
    # 3. Create splits file
    # Train: 8 CONTROL (1-8), 8 ASD (11-18)
    # Val: 2 CONTROL (9-10), 2 ASD (19-20)
    train_subs = [f"sub-ab{i:02d}" for i in list(range(1, 9)) + list(range(11, 19))]
    val_subs = [f"sub-ab{i:02d}" for i in [9, 10, 19, 20]]
    
    splits_data = {
        "train": train_subs,
        "val": val_subs,
        "test": []
    }
    split_file = tmp_path / "abide_splits.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=4)
        
    # 4. Write custom test preprocessor config limiting resize shape to 32x32x32 for fast CPU testing
    config_yaml = tmp_path / "test_preprocessing.yaml"
    with open(config_yaml, "w", encoding="utf-8") as f:
        f.write("""
cache_version: "2026-06-baseline"
profiles:
  autism:
    - transform: reorient
      params: { target: "RAS" }
    - transform: bias_correction
      params: { strategy: "n4", mode: "fast", convergence_threshold: 0.000001 }
    - transform: skull_strip
      params: { strategy: "threshold" }
    - transform: normalize
      params: { mode: "z_score" }
    - transform: crop_foreground
      params: {}
    - transform: resample
      params: { spacing: [1.0, 1.0, 1.0] }
    - transform: resize
      params: { target_shape: [32, 32, 32] }
""")
    
    # 5. Run the training experiment for 2 epochs
    run_training_experiment(
        data_root=raw_dir,
        index_file=index_file,
        split_file=split_file,
        preprocessed_dir=processed_dir,
        config_yaml=config_yaml,
        experiment_dir=experiment_dir,
        epochs=2,
        batch_size=2,
        device="cpu",
        lr=1e-3
    )
    
    # 6. Verify cached preprocessing files exist and are valid torch tensors
    for sub_id in subjects:
        cache_file = processed_dir / "2026-06-baseline" / f"{sub_id}.pt"
        assert cache_file.exists(), f"Preprocessed cache file {cache_file} should exist"
        
        # Load and verify content structure
        cache_data = torch.load(cache_file, map_location="cpu", weights_only=False)
        assert "image" in cache_data
        assert "affine" in cache_data
        assert "metadata" in cache_data
        
    # Verify experiment outputs
    assert (experiment_dir / "best_model.pt").exists()
    assert (experiment_dir / "latest_model.pt").exists()
    assert (experiment_dir / "best_model.onnx").exists()
    assert (experiment_dir / "roc_curve.png").exists()
    assert (experiment_dir / "confusion_matrix.png").exists()
    assert (experiment_dir / "classification_report.json").exists()
    assert (experiment_dir / "config.yaml").exists()
    assert (experiment_dir / "history.json").exists()
    assert (experiment_dir / "loss.png").exists()
    assert (experiment_dir / "accuracy.png").exists()
    
    # Verify new cache validation report and backups
    assert (experiment_dir / "cache_validation_report.json").exists()
    assert (experiment_dir / "preprocessing.yaml").exists()
    assert (experiment_dir / "cache_metadata.json").exists()
    
    with open(experiment_dir / "cache_validation_report.json", encoding="utf-8") as f:
        val_report = json.load(f)
    assert val_report["valid"] is True
    assert val_report["pipeline_hash_match"] is True
    assert val_report["subjects_checked"] == 20
    
    # Verify classification report content
    with open(experiment_dir / "classification_report.json", encoding="utf-8") as f:
        report = json.load(f)
    assert "Control" in report
    assert "Autism" in report
    
    # Verify configuration content
    with open(experiment_dir / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["epochs"] == 2
    assert config["batch_size"] == 2

    # 7. Verify training resumption behaves properly
    resume_experiment_dir = tmp_path / "experiment_resume"
    resume_experiment_dir.mkdir()
    
    # Run again resuming from the previous latest model
    run_training_experiment(
        data_root=raw_dir,
        index_file=index_file,
        split_file=split_file,
        preprocessed_dir=processed_dir,
        config_yaml=config_yaml,
        experiment_dir=resume_experiment_dir,
        epochs=3, # Train 1 more epoch (starts at 2, stops at 3)
        batch_size=2,
        device="cpu",
        lr=1e-3,
        resume_from=experiment_dir / "latest_model.pt"
    )
    
    # Check that resumption files exist in resume_experiment_dir
    assert (resume_experiment_dir / "latest_model.pt").exists()
    assert (resume_experiment_dir / "best_model.pt").exists()


def test_cache_validation_failure(tmp_path):
    """Test that cache validation properly detects mismatches and honors strict mode."""
    # 1. Create directories
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    experiment_dir = tmp_path / "experiment"
    
    # Generate 3 mock subjects to avoid Batch Normalization and ZeroDivisionError
    subjects = ["sub-ab01", "sub-ab02", "sub-ab03"]
    index_data = []
    for sub_id in subjects:
        sub_dir = raw_dir / "CONTROL" / sub_id
        sub_dir.mkdir(parents=True)
        img_path = sub_dir / "scan.nii.gz"
        data = np.random.randint(0, 100, size=(32, 32, 32), dtype=np.int16)
        nii_img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(nii_img, str(img_path))
        
        index_data.append({
            "subject_id": sub_id,
            "path": str(img_path),
            "label": "CONTROL"
        })
    
    index_file = tmp_path / "abide_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=4)
        
    splits_data = {
        "train": ["sub-ab01", "sub-ab02"],
        "val": ["sub-ab03"],
        "test": []
    }
    split_file = tmp_path / "abide_splits.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=4)
        
    config_yaml = tmp_path / "test_preprocessing.yaml"
    with open(config_yaml, "w", encoding="utf-8") as f:
        f.write("""
cache_version: "2026-06-baseline"
profiles:
  autism:
    - transform: reorient
      params: { target: "RAS" }
    - transform: resize
      params: { target_shape: [32, 32, 32] }
""")
        
    # Run preprocessing once to create valid cache
    from training.preprocess_autism import preprocess_abide_dataset
    preprocess_abide_dataset(index_file, processed_dir, config_yaml, raw_dir=raw_dir)
    
    for sub_id in subjects:
        cache_file = processed_dir / "2026-06-baseline" / f"{sub_id}.pt"
        assert cache_file.exists()
    
    # 2. Modify config to change target shape to 64x64x64, which will create shape and pipeline hash mismatch
    bad_config_yaml = tmp_path / "test_preprocessing_bad.yaml"
    with open(bad_config_yaml, "w", encoding="utf-8") as f:
        f.write("""
cache_version: "2026-06-baseline"
profiles:
  autism:
    - transform: reorient
      params: { target: "RAS" }
    - transform: resize
      params: { target_shape: [64, 64, 64] }
""")
        
    # If strict_cache_validation is True, running training should raise ValueError
    with pytest.raises(ValueError, match="Cache validation failed"):
        run_training_experiment(
            data_root=raw_dir,
            index_file=index_file,
            split_file=split_file,
            preprocessed_dir=processed_dir,
            config_yaml=bad_config_yaml,
            experiment_dir=experiment_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            lr=1e-3,
            skip_preprocess=True,  # Don't overwrite the bad cache
            strict_cache_validation=True
        )
        
    # If strict_cache_validation is False, it should warning and run through
    # (since epochs=1 and it's a small mock, it will run fine, albeit with warning logged)
    run_training_experiment(
        data_root=raw_dir,
        index_file=index_file,
        split_file=split_file,
        preprocessed_dir=processed_dir,
        config_yaml=bad_config_yaml,
        experiment_dir=experiment_dir,
        epochs=1,
        batch_size=2,
        device="cpu",
        lr=1e-3,
        skip_preprocess=True,
        strict_cache_validation=False
    )
    
    # Check that the report was written and is marked invalid
    report_path = experiment_dir / "cache_validation_report.json"
    assert report_path.exists()
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["valid"] is False
    assert report["pipeline_hash_match"] is False


def test_local_cache_copying(tmp_path):
    """Verify that train_autism.py correctly clones the remote cache directory to a local override folder."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    local_cache_dir = tmp_path / "local_cache"
    experiment_dir = tmp_path / "experiment"

    subjects = ["sub-ab01", "sub-ab02", "sub-ab03", "sub-ab04"]
    index_data = []

    for sub_id in subjects:
        sub_dir = raw_dir / "CONTROL" / sub_id
        sub_dir.mkdir(parents=True)
        img_path = sub_dir / "scan.nii.gz"
        data = np.random.randint(0, 100, size=(32, 32, 32), dtype=np.int16)
        nii_img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(nii_img, str(img_path))
        index_data.append({"subject_id": sub_id, "path": str(img_path), "label": "CONTROL"})

    index_file = tmp_path / "abide_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=4)

    splits_data = {"train": ["sub-ab01", "sub-ab02"], "val": ["sub-ab03", "sub-ab04"], "test": []}
    split_file = tmp_path / "abide_splits.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=4)

    config_yaml = tmp_path / "test_preprocessing.yaml"
    with open(config_yaml, "w", encoding="utf-8") as f:
        f.write("""
cache_version: "2026-06-baseline"
profiles:
  autism:
    - transform: reorient
      params: { target: "RAS" }
    - transform: resize
      params: { target_shape: [32, 32, 32] }
""")

    # Run training with --local-cache-dir, which triggers preprocessing and copying to local_cache
    run_training_experiment(
        data_root=raw_dir,
        index_file=index_file,
        split_file=split_file,
        preprocessed_dir=processed_dir,
        config_yaml=config_yaml,
        experiment_dir=experiment_dir,
        epochs=1,
        batch_size=2,
        device="cpu",
        lr=1e-3,
        local_cache_dir=local_cache_dir,
    )

    # Check that the files were copied to local_cache_dir under the correct cache version
    local_version_dir = local_cache_dir / "2026-06-baseline"
    assert local_version_dir.exists()
    assert (local_version_dir / "metadata.json").exists()
    for sub_id in subjects:
        assert (local_version_dir / f"{sub_id}.pt").exists()
