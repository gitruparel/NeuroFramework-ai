import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from training.evaluation import assemble_oof_predictions, tune_optimal_thresholds, generate_cv_report
from training.train_autism import run_training_experiment


def test_tune_optimal_thresholds(tmp_path):
    """Verify that threshold optimization locates correct thresholds and outputs JSON files."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.95, 0.3, 0.85, 0.15, 0.9])
    
    thresholds = tune_optimal_thresholds(y_true, y_prob, tmp_path)
    
    json_path = tmp_path / "optimal_thresholds.json"
    assert json_path.exists()
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert "best_f1_threshold" in data
    assert "best_balanced_accuracy_threshold" in data
    assert "best_youden_threshold" in data
    assert data["best_f1_value"] == 1.0  # Perfect prediction split should hit 1.0 F1


def test_oof_predictions_assembly(tmp_path):
    """Verify assembly of predictions.csv files across multiple folds."""
    # Create fold folders and prediction files
    for f_idx in range(5):
        fold_dir = tmp_path / f"fold_{f_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            "Subject": [f"sub_{f_idx}_0", f"sub_{f_idx}_1"],
            "True_Label": ["Control", "Autism"],
            "Pred_Label": ["Control", "Autism"],
            "Probability_ASD": [0.1, 0.9],
            "Probability_Control": [0.9, 0.1],
            "Logit_ASD": [-2.0, 2.0],
            "Logit_Control": [2.0, -2.0],
            "Correct": [True, True]
        })
        df.to_csv(fold_dir / "predictions.csv", index=False)
        
    oof_df = assemble_oof_predictions(tmp_path, n_folds=5)
    
    oof_csv_path = tmp_path / "oof_predictions.csv"
    assert oof_csv_path.exists()
    assert len(oof_df) == 10
    assert "Fold" in oof_df.columns
    assert sorted(oof_df["Fold"].unique().tolist()) == [0, 1, 2, 3, 4]


def test_generate_cv_report(tmp_path):
    """Verify CV summary CSV and PNG generation from fold metadata files."""
    for f_idx in range(5):
        fold_dir = tmp_path / f"fold_{f_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        meta = {
            "best_val_accuracy": 0.70 + (f_idx * 0.02),
            "best_val_roc_auc": 0.75 + (f_idx * 0.01),
            "best_val_pr_auc": 0.72 + (f_idx * 0.015),
            "best_val_macro_f1": 0.68 + (f_idx * 0.02),
            "best_val_balanced_accuracy": 0.69 + (f_idx * 0.02),
            "best_val_sensitivity": 0.65 + (f_idx * 0.03),
            "best_val_specificity": 0.73 + (f_idx * 0.01)
        }
        with open(fold_dir / "experiment_meta.json", "w") as f:
            json.dump(meta, f)
            
    generate_cv_report(tmp_path, n_folds=5)
    
    csv_path = tmp_path / "cv_summary.csv"
    png_path = tmp_path / "cv_summary.png"
    
    assert csv_path.exists()
    assert png_path.exists()
    
    df = pd.read_csv(csv_path)
    # 5 folds + 1 Mean + 1 Std = 7 rows
    assert len(df) == 7
    assert df.loc[df["Fold"] == "Mean", "Accuracy"].values[0] == pytest.approx(0.74, 1e-5)


def test_cv_cli_smoke_test(tmp_path):
    """Verify that train_autism.py --cv runs end-to-end sequentially over 5 folds and generates outputs."""
    import sys
    import subprocess
    import nibabel as nib
    
    # 1. Setup mock subject directories
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    experiment_dir = tmp_path / "experiment"
    
    subjects = []
    index_data = []
    
    # 8 subjects total (4 CONTROL, 4 ASD)
    for i in range(1, 9):
        sub_id = f"sub-ab{i:02d}"
        label = "CONTROL" if i <= 4 else "ASD"
        sub_dir = raw_dir / label / sub_id
        sub_dir.mkdir(parents=True)
        
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
        
    # 2. Setup mock split file
    splits_file = tmp_path / "abide_splits.json"
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump({"train": subjects[:6], "val": subjects[6:]}, f, indent=4)
        
    # 3. Setup mock k-fold file (5 folds)
    # Each fold: 6 training subjects, 2 validation subjects
    kfold_data = []
    for fold_idx in range(5):
        # Shift validation subjects per fold
        val_indices = [(fold_idx * 2 + j) % 8 for j in range(2)]
        train_indices = [idx for idx in range(8) if idx not in val_indices]
        
        kfold_data.append({
            "train": [subjects[idx] for idx in train_indices],
            "val": [subjects[idx] for idx in val_indices]
        })
        
    kfold_file = tmp_path / "abide_kfold.json"
    with open(kfold_file, "w", encoding="utf-8") as f:
        json.dump(kfold_data, f, indent=4)
        
    # 4. Write fast CPU preprocessing YAML
    config_yaml = tmp_path / "test_preprocessing.yaml"
    with open(config_yaml, "w", encoding="utf-8") as f:
        f.write("""
cache_version: "2026-06-cv"
profiles:
  autism:
    - transform: reorient
      params: { target: "RAS" }
    - transform: resize
      params: { target_shape: [32, 32, 32] }
""")

    # 5. Run preprocessing once locally via run_training_experiment to generate the cache
    run_training_experiment(
        data_root=raw_dir,
        index_file=index_file,
        split_file=splits_file,
        preprocessed_dir=processed_dir,
        config_yaml=config_yaml,
        experiment_dir=tmp_path / "init_exp",
        epochs=1,
        batch_size=2,
        device="cpu",
        lr=1e-3
    )

    # 6. Execute the full --cv training sweep using subprocess
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "training.train_autism",
        "--data-root", str(raw_dir),
        "--index-file", str(index_file),
        "--split-file", str(splits_file),
        "--preprocessed-dir", str(processed_dir),
        "--config-yaml", str(config_yaml),
        "--experiment-dir", str(experiment_dir),
        "--epochs", "1",
        "--batch-size", "2",
        "--device", "cpu",
        "--skip-preprocess",
        "--kfold-file", str(kfold_file),
        "--cv"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CV execution failed: {result.stderr}"
    
    # 7. Validate that all folds trained and output reports were generated
    for fold_idx in range(5):
        fold_dir = experiment_dir / f"fold_{fold_idx}"
        assert fold_dir.exists(), f"Directory for fold {fold_idx} should exist"
        assert (fold_dir / "predictions.csv").exists()
        assert (fold_dir / "experiment_meta.json").exists()
        
    assert (experiment_dir / "oof_predictions.csv").exists()
    assert (experiment_dir / "calibration_curve.png").exists()
    assert (experiment_dir / "optimal_thresholds.json").exists()
    assert (experiment_dir / "cv_summary.csv").exists()
    assert (experiment_dir / "cv_summary.png").exists()

