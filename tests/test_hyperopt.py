"""Unit and integration tests for the hyperparameter optimization module."""

import json
import csv
from pathlib import Path
import pytest
import numpy as np
import nibabel as nib

from training.hyperopt import optimize_hyperparameters
from training.train_autism import run_training_experiment


def test_hyperopt_engine_unit_test(tmp_path):
    """Verify that optimize_hyperparameters logs metrics, CSVs, and plots correctly."""
    # 1. Define dummy objective function simulating suggestions
    def dummy_objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        dropout = trial.suggest_float("dropout_prob", 0.0, 0.5)
        # Objective function score
        return float(np.sin(lr) + batch_size * 0.01 - dropout)
        
    output_dir = tmp_path / "hyperopt_outputs"
    
    # 2. Run the optimization engine
    best_results = optimize_hyperparameters(
        objective_fn=dummy_objective,
        n_trials=5,
        seed=42,
        output_dir=output_dir
    )
    
    # 3. Assert outputs are created
    assert (output_dir / "optuna_best.json").exists()
    assert (output_dir / "optuna_trials.csv").exists()
    assert (output_dir / "optimization_history.png").exists()
    assert (output_dir / "parameter_importance.png").exists()
    
    # 4. Verify JSON content
    with open(output_dir / "optuna_best.json", "r", encoding="utf-8") as f:
        best_data = json.load(f)
    assert "best_trial_number" in best_data
    assert "best_value" in best_data
    assert "best_params" in best_data
    assert best_data["best_params"]["batch_size"] in [8, 16]
    
    # 5. Verify CSV content
    with open(output_dir / "optuna_trials.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 5
    for r in rows:
        assert "trial_number" in r
        assert "state" in r
        assert "value" in r
        assert "lr" in r
        assert "batch_size" in r
        assert "dropout_prob" in r


def test_autism_hyperopt_integration(tmp_path):
    """Verify end-to-end train_autism.py integration with Optuna hyperparameter optimization."""
    # 1. Create temporary mock folders
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    experiment_dir = tmp_path / "experiment"
    
    # 2. Generate small mock dataset
    np.random.seed(42)
    index_data = []
    for i in range(1, 11):
        sub_id = f"sub-mock{i:02d}"
        label = "CONTROL" if i <= 5 else "ASD"
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
        
    index_file = tmp_path / "abide_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=4)
        
    # Create splits
    train_subs = [f"sub-mock{i:02d}" for i in [1, 2, 3, 6, 7, 8]]
    val_subs = [f"sub-mock{i:02d}" for i in [4, 5, 9, 10]]
    
    splits_data = {
        "train": train_subs,
        "val": val_subs,
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

    # 3. Trigger Optuna study with 2 trials and 1 epoch per trial
    # Wrapping inside a dummy function that parses options similar to the main block
    from training.train_autism import run_training_experiment
    from training.hyperopt import optimize_hyperparameters
    import shutil
    
    def objective(trial):
        trial_lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        trial_weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
        trial_batch_size = trial.suggest_categorical("batch_size", [2, 4])
        trial_dropout = trial.suggest_float("dropout_prob", 0.0, 0.2)
        trial_patience = trial.suggest_int("scheduler_patience", 3, 5)
        trial_factor = trial.suggest_float("scheduler_factor", 0.3, 0.5)
        
        trial_dir = experiment_dir / f"trial_{trial.number}"
        
        metrics = run_training_experiment(
            data_root=raw_dir,
            index_file=index_file,
            split_file=split_file,
            preprocessed_dir=processed_dir,
            config_yaml=config_yaml,
            experiment_dir=trial_dir,
            epochs=1,
            batch_size=trial_batch_size,
            device="cpu",
            lr=trial_lr,
            resume_from="none",
            skip_preprocess=False,
            copy_outputs_to=None,
            strict_cache_validation=True,
            limit=None,
            full_cache_validation=False,
            local_cache_dir=None,
            augment=False,
            optimizer_name="adam",
            weight_decay=trial_weight_decay,
            scheduler_patience=trial_patience,
            scheduler_factor=trial_factor,
            seed=42,
            experiment_name=f"test_trial_{trial.number}",
            decision_threshold=0.5,
            early_stopping_patience=5,
            dropout_prob=trial_dropout,
            label_smoothing=0.0,
            use_class_weights=False,
            is_hyperopt=True,
        )
        
        # Ensure cleanup happened inside objective if we want, or do it in try-finally
        shutil.rmtree(trial_dir)
        return metrics.get("best_val_accuracy", 0.0)

    # Run search
    best_results = optimize_hyperparameters(
        objective_fn=objective,
        n_trials=2,
        seed=42,
        output_dir=experiment_dir
    )
    
    # 4. Verify hyperparameter search outputs
    assert (experiment_dir / "optuna_best.json").exists()
    assert (experiment_dir / "optuna_trials.csv").exists()
    assert (experiment_dir / "optimization_history.png").exists()
    assert (experiment_dir / "parameter_importance.png").exists()
    
    # Verify final training run executes successfully using best hyperparams
    best_params = best_results["best_params"]
    run_training_experiment(
        data_root=raw_dir,
        index_file=index_file,
        split_file=split_file,
        preprocessed_dir=processed_dir,
        config_yaml=config_yaml,
        experiment_dir=experiment_dir,
        epochs=1,
        batch_size=best_params["batch_size"],
        device="cpu",
        lr=best_params["lr"],
        resume_from="none",
        skip_preprocess=True,
        copy_outputs_to=None,
        strict_cache_validation=True,
        limit=None,
        full_cache_validation=False,
        local_cache_dir=None,
        augment=False,
        optimizer_name="adam",
        weight_decay=best_params["weight_decay"],
        scheduler_patience=best_params["scheduler_patience"],
        scheduler_factor=best_params["scheduler_factor"],
        seed=42,
        experiment_name="test_final_run",
        decision_threshold=0.5,
        early_stopping_patience=5,
        dropout_prob=best_params["dropout_prob"],
        label_smoothing=0.0,
        use_class_weights=False,
        is_hyperopt=False,
    )
    
    # 5. Verify final output files exist in main experiment directory
    assert (experiment_dir / "best_model.onnx").exists()
    assert (experiment_dir / "predictions.csv").exists()
    assert (experiment_dir / "classification_report.json").exists()
