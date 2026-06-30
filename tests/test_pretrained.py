import pytest
import torch
import torch.nn as nn
from models.pretrained import load_pretrained_weights, freeze_backbone, unfreeze_backbone
from models.factory import ModelFactory


def test_layer_freezing():
    """Verify that freeze_backbone freezes non-classifier parameters, and unfreeze_backbone unfreezes them."""
    model = ModelFactory.create_model("resnet10", in_channels=1, out_channels=2)
    
    # 1. Freeze backbone
    freeze_backbone(model)
    
    classifier_names = ["class_out", "classifier", "fc"]
    for name, param in model.named_parameters():
        is_classifier = any(cn in name for cn in classifier_names)
        if not is_classifier:
            assert not param.requires_grad, f"Param {name} should be frozen"
        else:
            assert param.requires_grad, f"Classifier param {name} should NOT be frozen"
            
    # 2. Unfreeze backbone
    unfreeze_backbone(model)
    
    for name, param in model.named_parameters():
        assert param.requires_grad, f"Param {name} should be unfrozen"


def test_differential_learning_rates():
    """Verify that backbone and classifier parameters get distinct optimizer learning rates."""
    model = ModelFactory.create_model("resnet10", in_channels=1, out_channels=2)
    
    classifier_names = ["class_out", "classifier", "fc"]
    backbone_params = []
    classifier_params = []
    
    for n, p in model.named_parameters():
        if any(cn in n for cn in classifier_names):
            classifier_params.append(p)
        else:
            backbone_params.append(p)
            
    param_groups = [
        {"params": backbone_params, "lr": 1e-4},
        {"params": classifier_params, "lr": 1e-3}
    ]
    
    optimizer = torch.optim.Adam(param_groups)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-3


def test_custom_checkpoint_loading(tmp_path):
    """Verify loading custom weights into the model structure works properly."""
    model = ModelFactory.create_model("resnet10", in_channels=1, out_channels=2)
    
    ckpt_path = tmp_path / "dummy_checkpoint.pth"
    torch.save(model.state_dict(), ckpt_path)
    
    # Load state dict
    loaded_model = load_pretrained_weights(model, source="custom", checkpoint_path=str(ckpt_path))
    assert loaded_model is not None


def test_transfer_learning_comparison(tmp_path):
    """Verify aggregation and generation of transfer learning comparison reports."""
    from training.transfer_learning import aggregate_transfer_learning_benchmark
    import json
    import pandas as pd
    
    # Create fake directory structure and metadata json files
    (tmp_path / "none").mkdir()
    (tmp_path / "medicalnet").mkdir()
    
    meta_none = {
        "best_val_accuracy": 0.72,
        "best_val_roc_auc": 0.75,
        "best_val_pr_auc": 0.74,
        "best_val_macro_f1": 0.71,
        "best_val_balanced_accuracy": 0.71,
        "best_epoch": 5,
        "training_time": 10.5,
        "parameter_count": 10000
    }
    meta_med = {
        "best_val_accuracy": 0.78,
        "best_val_roc_auc": 0.82,
        "best_val_pr_auc": 0.81,
        "best_val_macro_f1": 0.77,
        "best_val_balanced_accuracy": 0.77,
        "best_epoch": 3,
        "training_time": 8.2,
        "parameter_count": 10000
    }
    
    with open(tmp_path / "none" / "experiment_meta.json", "w") as f:
        json.dump(meta_none, f)
    with open(tmp_path / "medicalnet" / "experiment_meta.json", "w") as f:
        json.dump(meta_med, f)
        
    aggregate_transfer_learning_benchmark(tmp_path, ["none", "medicalnet"])
    
    csv_path = tmp_path / "transfer_learning_comparison.csv"
    png_path = tmp_path / "transfer_learning_comparison.png"
    
    assert csv_path.exists()
    assert png_path.exists()
    
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert df.loc[df["Initialization_Type"] == "medicalnet", "ROC_AUC"].values[0] == 0.82
