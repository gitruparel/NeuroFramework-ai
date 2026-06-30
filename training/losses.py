"""Unified loss functions and factory for medical image classification."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.logging import setup_logger

logger = setup_logger("training.losses", "training/losses.log")


class FocalLoss(nn.Module):
    """Vectorized Focal Loss implementation with Label Smoothing and Class Weighting support.
    
    Supports soft smoothing distribution and binary/multi-class alpha weight mapping.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Convert targets to one-hot representation if they are class indices
        if targets.dim() == 1 or targets.size(-1) == 1:
            targets_idx = targets.view(-1).long()
            one_hot = torch.zeros_like(logits).scatter_(1, targets_idx.view(-1, 1), 1.0)
        else:
            one_hot = targets.float()

        # Apply label smoothing if configured
        if self.label_smoothing > 0.0:
            one_hot = one_hot * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        # Basic cross entropy terms per class
        ce = -one_hot * log_p

        # Focal factor: (1 - p)^gamma
        focal_factor = torch.pow(1.0 - p, self.gamma)
        loss = focal_factor * ce

        # Apply alpha class weights if present
        if self.alpha is not None:
            alpha_device = self.alpha.to(loss.device)
            loss = loss * alpha_device

        # Sum losses across classes
        loss = loss.sum(dim=-1)

        # Reduce across batch elements
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LossFactory:
    """Factory responsible for constructing PyTorch losses for training."""

    @staticmethod
    def create_loss(
        loss_name: str,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0
    ) -> nn.Module:
        """Constructs a standard or advanced medical loss criterion."""
        name_lower = loss_name.lower().replace("-", "").replace("_", "")
        
        if name_lower == "ce":
            return nn.CrossEntropyLoss(reduction="mean")
        elif name_lower == "weightedce":
            if alpha is None:
                raise ValueError("Class weights (alpha) must be provided for weighted CrossEntropyLoss.")
            return nn.CrossEntropyLoss(weight=alpha, reduction="mean")
        elif name_lower == "cels":
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="mean")
        elif name_lower == "focal":
            return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        elif name_lower == "focalls":
            return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean", label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown loss function name: {loss_name}")


def generate_loss_plots(csv_path: Path, output_img_path: Path) -> None:
    """Generates comparison bar plots ranking performance metrics across loss functions."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if not csv_path.exists():
        logger.error(f"Cannot generate loss comparison plots. CSV not found: {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read loss comparison CSV: {e}")
        return
        
    plt.close("all")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Loss Function Benchmark Comparison", fontsize=16, fontweight="bold")
    
    metrics = [
        ("Val_Accuracy", "Validation Accuracy", "royalblue"),
        ("ROC_AUC", "ROC-AUC Score", "seagreen"),
        ("PR_AUC", "PR-AUC Score", "orange"),
        ("F1", "Macro F1-Score", "purple"),
        ("Balanced_Accuracy", "Balanced Accuracy", "crimson")
    ]
    
    # 1. Plot the standard 5 metrics
    for idx, (col, title, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Metric '{col}' not found", ha="center", va="center")
            ax.axis("off")
            continue
            
        x_vals = df["Loss_Function"]
        y_vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        ax.bar(x_vals, y_vals, color=color, edgecolor="black", alpha=0.85, width=0.4)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.1)
        
        for i, val in enumerate(y_vals):
            ax.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
            
    # 2. Plot Grouped Sensitivity vs Specificity bar chart in the 6th slot
    ax = axes[1, 2]
    if "Sensitivity" in df.columns and "Specificity" in df.columns:
        x = np.arange(len(df["Loss_Function"]))
        width = 0.3
        
        sens_vals = pd.to_numeric(df["Sensitivity"], errors='coerce').fillna(0.0)
        spec_vals = pd.to_numeric(df["Specificity"], errors='coerce').fillna(0.0)
        
        ax.bar(x - width/2, sens_vals, width, label='Sensitivity', color='teal', edgecolor='black', alpha=0.85)
        ax.bar(x + width/2, spec_vals, width, label='Specificity', color='coral', edgecolor='black', alpha=0.85)
        
        ax.set_title("Sensitivity vs Specificity", fontsize=12, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Loss_Function"])
        ax.legend(loc='lower left')
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.1)
        
        for i in range(len(x)):
            ax.text(i - width/2, sens_vals[i] + 0.01, f"{sens_vals[i]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width/2, spec_vals[i] + 0.01, f"{spec_vals[i]:.3f}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Sensitivity/Specificity not found", ha="center", va="center")
        ax.axis("off")
        
    plt.tight_layout()
    try:
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved loss comparison plots to: {output_img_path}")
    except Exception as e:
        logger.error(f"Failed to save loss comparison plots: {e}")
    finally:
        plt.close("all")


def aggregate_losses_benchmark(experiment_dir: Path, loss_names: List[str]) -> None:
    """Aggregates metadata across loss experiment folders, compiling loss_comparison.csv and PNG plots."""
    records = []
    for loss in loss_names:
        meta_path = experiment_dir / loss / "experiment_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    
                record = {
                    "Loss_Function": loss,
                    "Val_Accuracy": meta.get("best_val_accuracy", 0.0),
                    "ROC_AUC": meta.get("best_val_roc_auc", 0.0),
                    "PR_AUC": meta.get("best_val_pr_auc", 0.0),
                    "F1": meta.get("best_val_macro_f1", meta.get("macro_f1", 0.0)),
                    "Balanced_Accuracy": meta.get("best_val_balanced_accuracy", 0.0),
                    "Sensitivity": meta.get("best_val_sensitivity", 0.0),
                    "Specificity": meta.get("best_val_specificity", 0.0),
                    "Best_Epoch": meta.get("best_epoch", 0),
                    "Training_Time": meta.get("training_time", 0.0),
                }
                records.append(record)
            except Exception as e:
                logger.error(f"Failed to read metadata for loss {loss} during aggregation: {e}")
                
    if not records:
        logger.warning("No experiment metadata found to aggregate for loss functions.")
        return
        
    import pandas as pd
    csv_path = experiment_dir / "loss_comparison.csv"
    try:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved loss comparison table to: {csv_path}")
        
        plot_path = experiment_dir / "loss_comparison.png"
        generate_loss_plots(csv_path, plot_path)
    except Exception as e:
        logger.error(f"Failed to save loss comparison CSV: {e}")
