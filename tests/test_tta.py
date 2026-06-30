import pytest
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from training.inference import TestTimeAugmentor, PredictionAggregator, generate_tta_comparison_plots, aggregate_tta_comparison


def test_tta_augmentor_shape_and_determinism():
    """Verify that TestTimeAugmentor output shape is preserved and execution is deterministic."""
    augmentor = TestTimeAugmentor(seed=42)
    # Shape (C, D, H, W)
    dummy_scan = torch.randn(1, 16, 16, 16)
    
    # 1. Check all 5 default TTA runs preserve shape
    for run_idx in range(5):
        aug_scan = augmentor.get_augmentations(dummy_scan, run_idx)
        assert aug_scan.shape == dummy_scan.shape, f"Shape mismatch for run {run_idx}"
        
    # 2. Check deterministic replication
    scan_1 = augmentor.get_augmentations(dummy_scan, 2)
    scan_2 = augmentor.get_augmentations(dummy_scan, 2)
    assert torch.allclose(scan_1, scan_2), "TTA spatial transformation not deterministic"
    
    # 3. Check randomized run (run_idx >= 5) deterministic seeding
    aug_r5_1 = augmentor.get_augmentations(dummy_scan, 5)
    aug_r5_2 = augmentor.get_augmentations(dummy_scan, 5)
    assert torch.allclose(aug_r5_1, aug_r5_2), "Random TTA run not deterministic across calls with identical seed"


def test_prediction_aggregator():
    """Verify prediction probability aggregation strategies (mean, median, majority)."""
    # 3 runs, batch size 2, class count 2
    probs_list = [
        torch.tensor([[0.2, 0.8], [0.9, 0.1]]), # run 0
        torch.tensor([[0.4, 0.6], [0.7, 0.3]]), # run 1
        torch.tensor([[0.3, 0.7], [0.2, 0.8]]), # run 2
    ]
    
    # 1. Mean aggregation
    mean_agg = PredictionAggregator.aggregate(probs_list, method="mean")
    expected_mean = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    assert torch.allclose(mean_agg, expected_mean, atol=1e-5)
    
    # 2. Median aggregation
    median_agg = PredictionAggregator.aggregate(probs_list, method="median")
    expected_median = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
    assert torch.allclose(median_agg, expected_median, atol=1e-5)
    
    # 3. Majority aggregation
    # Class votes:
    # Batch 0: Class 1 (0.8), Class 1 (0.6), Class 1 (0.7) -> Majority Class 1
    # Batch 1: Class 0 (0.9), Class 0 (0.7), Class 1 (0.8) -> Majority Class 0
    majority_agg = PredictionAggregator.aggregate(probs_list, method="majority")
    expected_majority = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(majority_agg, expected_majority, atol=1e-5)


def test_prediction_aggregator_invalid_method():
    """Verify that PredictionAggregator raises ValueError for invalid methods."""
    probs_list = [torch.randn(2, 2)]
    with pytest.raises(ValueError, match="Unknown TTA aggregation method"):
        PredictionAggregator.aggregate(probs_list, method="invalid")


def test_tta_comparison_reports(tmp_path):
    """Verify that aggregate_tta_comparison compiles valid CSVs and PNG comparison plots."""
    baseline_metrics = {
        "accuracy": 0.70,
        "roc_auc": 0.74,
        "pr_auc": 0.72,
        "macro_f1": 0.68,
        "balanced_accuracy": 0.69,
        "sensitivity": 0.65,
        "specificity": 0.73
    }
    tta_metrics = {
        "accuracy": 0.75,
        "roc_auc": 0.81,
        "pr_auc": 0.79,
        "macro_f1": 0.74,
        "balanced_accuracy": 0.74,
        "sensitivity": 0.70,
        "specificity": 0.78
    }
    
    aggregate_tta_comparison(
        baseline_metrics=baseline_metrics,
        tta_metrics=tta_metrics,
        baseline_latency=1.5,
        tta_latency=6.8,
        output_dir=tmp_path
    )
    
    csv_path = tmp_path / "tta_comparison.csv"
    png_path = tmp_path / "tta_comparison.png"
    
    assert csv_path.exists()
    assert png_path.exists()
    
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert df.loc[df["Mode"] == "TTA", "ROC_AUC"].values[0] == 0.81
    assert df.loc[df["Mode"] == "Baseline", "Latency_sec"].values[0] == 1.5
