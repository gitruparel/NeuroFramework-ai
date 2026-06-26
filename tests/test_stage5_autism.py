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
        
    # 4. Use existing configs/preprocessing.yaml
    config_yaml = Path("configs/preprocessing.yaml")
    assert config_yaml.exists(), "configs/preprocessing.yaml must exist"
    
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
        cache_file = processed_dir / f"{sub_id}.pt"
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
