import pytest
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from training.losses import FocalLoss, LossFactory, generate_loss_plots, aggregate_losses_benchmark


def test_loss_factory_initialization():
    """Verify that LossFactory builds all loss functions with appropriate shapes."""
    alpha = torch.tensor([0.25, 0.75], dtype=torch.float)
    
    # 1. Standard Cross Entropy
    ce_loss = LossFactory.create_loss("ce")
    assert isinstance(ce_loss, nn.CrossEntropyLoss)
    assert ce_loss.weight is None
    
    # 2. Weighted Cross Entropy
    weighted_ce = LossFactory.create_loss("weighted_ce", alpha=alpha)
    assert isinstance(weighted_ce, nn.CrossEntropyLoss)
    assert torch.equal(weighted_ce.weight, alpha)
    
    # 3. Label smoothed Cross Entropy
    ce_ls = LossFactory.create_loss("ce_ls", label_smoothing=0.1)
    assert isinstance(ce_ls, nn.CrossEntropyLoss)
    assert ce_ls.label_smoothing == 0.1
    
    # 4. Focal Loss
    focal = LossFactory.create_loss("focal", alpha=alpha, gamma=3.0)
    assert isinstance(focal, FocalLoss)
    assert torch.equal(focal.alpha, alpha)
    assert focal.gamma == 3.0
    assert focal.label_smoothing == 0.0
    
    # 5. Focal Loss with Label Smoothing
    focal_ls = LossFactory.create_loss("focal_ls", alpha=alpha, gamma=2.0, label_smoothing=0.15)
    assert isinstance(focal_ls, FocalLoss)
    assert torch.equal(focal_ls.alpha, alpha)
    assert focal_ls.gamma == 2.0
    assert focal_ls.label_smoothing == 0.15


def test_focal_loss_forward_and_backward():
    """Verify forward execution shape and gradient backpropagation for FocalLoss."""
    # Build a simple linear classification model
    model = nn.Linear(10, 2)
    inputs = torch.randn(4, 10)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    alpha = torch.tensor([0.4, 0.6], dtype=torch.float)
    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.1)
    
    logits = model(inputs)
    loss = criterion(logits, targets)
    
    # Assert output properties
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0
    
    # Backpropagation check
    loss.backward()
    assert model.weight.grad is not None
    assert model.bias.grad is not None
    assert not torch.isnan(model.weight.grad).any()


def test_loss_factory_invalid_inputs():
    """Verify that LossFactory handles errors correctly on wrong parameters."""
    with pytest.raises(ValueError, match="Class weights.*must be provided"):
        LossFactory.create_loss("weighted_ce", alpha=None)
        
    with pytest.raises(ValueError, match="Unknown loss function"):
        LossFactory.create_loss("invalid_loss_name")


def test_losses_benchmark_plotting_and_aggregation(tmp_path):
    """Verify that aggregate_losses_benchmark runs and compiles valid CSVs and PNGs."""
    # Create fake subfolders with experiment_meta.json files
    losses = ["ce", "weighted_ce", "focal", "ce_ls", "focal_ls"]
    
    for loss in losses:
        loss_dir = tmp_path / loss
        loss_dir.mkdir()
        meta = {
            "best_epoch": 5,
            "best_val_loss": 0.45,
            "best_val_accuracy": 0.78,
            "best_val_pr_auc": 0.82,
            "best_val_roc_auc": 0.85,
            "best_val_balanced_accuracy": 0.76,
            "best_val_macro_f1": 0.77,
            "best_val_sensitivity": 0.75,
            "best_val_specificity": 0.80,
            "training_time": 120.5,
        }
        import json
        with open(loss_dir / "experiment_meta.json", "w") as f:
            json.dump(meta, f)
            
    aggregate_losses_benchmark(tmp_path, losses)
    
    # Verify files are written
    csv_path = tmp_path / "loss_comparison.csv"
    png_path = tmp_path / "loss_comparison.png"
    
    assert csv_path.exists()
    assert png_path.exists()
    
    df = pd.read_csv(csv_path)
    assert len(df) == len(losses)
    assert "Loss_Function" in df.columns
    assert "Sensitivity" in df.columns
    assert "Specificity" in df.columns
    assert df.loc[df["Loss_Function"] == "focal", "ROC_AUC"].values[0] == 0.85
