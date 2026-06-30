"""Test-Time Augmentation (TTA) and inference utility functions."""

import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai.transforms as mt
from core.logging import setup_logger

logger = setup_logger("training.inference", "training/inference.log")


class TestTimeAugmentor:
    """Generates anatomically realistic augmented 3D scans for robust prediction."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_augmentations(self, image: torch.Tensor, run_idx: int) -> torch.Tensor:
        """Applies deterministic spatial or intensity transforms depending on run index."""
        device = image.device
        
        if run_idx == 0:
            # 1. Original image
            return image
        elif run_idx == 1:
            # 2. Left-Right Flip (RAS layout: lateral axis is typically the first spatial index, e.g. dims=[1] for (C, D, H, W))
            return torch.flip(image, dims=[1])
        elif run_idx == 2:
            # 3. Small Affine (mild rotation and translation)
            angle = 5.0 * (math.pi / 180.0)  # 5 degrees rotation
            affine_t = mt.Affine(
                rotate_params=[angle, 0.0, 0.0],
                translate_params=[1.0, 0.0, 0.0],
                padding_mode="zeros",
                device=device
            )
            res_affine = affine_t(image)
            return res_affine[0] if isinstance(res_affine, tuple) else res_affine
        elif run_idx == 3:
            # 4. Mild Intensity Scaling
            # b_min/b_max scales intensities relative to the percentile boundaries
            scale_t = mt.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0.0, b_max=1.05)
            return scale_t(image)
        elif run_idx == 4:
            # 5. Mild Contrast Adjustment
            contrast_t = mt.AdjustContrast(gamma=1.15)
            return contrast_t(image)
        else:
            # For runs >= 5: apply randomized combinations using a seed-controlled generator
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed + run_idx)
            
            out = image
            # Random lateral flip
            if torch.rand(1, generator=generator).item() > 0.5:
                out = torch.flip(out, dims=[1])
                
            # Random rotation/translation
            rot = (torch.rand(3, generator=generator) * 0.1 - 0.05).tolist()  # ±3 degrees
            trans = (torch.rand(3, generator=generator) * 1.0 - 0.5).tolist()  # ±0.5 voxel
            affine_t = mt.Affine(rotate_params=rot, translate_params=trans, padding_mode="zeros", device=device)
            res_affine = affine_t(out)
            out = res_affine[0] if isinstance(res_affine, tuple) else res_affine
            
            # Random contrast gamma adjustment
            gamma = (torch.rand(1, generator=generator).item() * 0.3 + 0.85)  # 0.85 to 1.15
            contrast_t = mt.AdjustContrast(gamma=gamma)
            out = contrast_t(out)
            
            return out


class PredictionAggregator:
    """Aggregates multiple probability outputs across runs."""

    @staticmethod
    def aggregate(probs_list: List[torch.Tensor], method: str = "mean") -> torch.Tensor:
        """Combines a list of prediction probability tensors (N, C) into a single prediction tensor."""
        stacked = torch.stack(probs_list, dim=0)  # (runs, N, C)
        
        if method == "mean":
            return torch.mean(stacked, dim=0)
        elif method == "median":
            return torch.median(stacked, dim=0).values
        elif method == "majority":
            # Predictions (argmax class for each run)
            preds = torch.argmax(stacked, dim=-1)  # (runs, N)
            # Find the most common class index along runs dim
            mode_res = torch.mode(preds, dim=0)
            majority_classes = mode_res.values  # (N,)
            
            # Convert to pseudo-probability (one-hot)
            num_classes = stacked.size(-1)
            majority_probs = F.one_hot(majority_classes, num_classes=num_classes).float()
            return majority_probs
        else:
            raise ValueError(f"Unknown TTA aggregation method: {method}")


def generate_tta_comparison_plots(csv_path: Path, output_img_path: Path) -> None:
    """Generates grouped bar charts comparing performance metrics and latency with vs without TTA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if not csv_path.exists():
        logger.error(f"Cannot generate TTA comparison plots. CSV not found: {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read TTA comparison CSV: {e}")
        return
        
    plt.close("all")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Test-Time Augmentation (TTA) Performance Evaluation", fontsize=16, fontweight="bold")
    
    metrics = [
        ("Accuracy", "Overall Accuracy", "royalblue"),
        ("ROC_AUC", "ROC-AUC Score", "seagreen"),
        ("PR_AUC", "PR-AUC Score", "orange"),
        ("Macro_F1", "Macro F1-Score", "purple"),
        ("Balanced_Accuracy", "Balanced Accuracy", "crimson")
    ]
    
    if "Mode" not in df.columns:
        logger.error("TTA CSV missing 'Mode' column.")
        return
        
    modes = df["Mode"].tolist()  # e.g., ["Baseline", "TTA"]
    
    # 1. Plot the standard 5 validation metrics
    for idx, (col, title, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Metric '{col}' not found", ha="center", va="center")
            ax.axis("off")
            continue
            
        y_vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0).tolist()
        
        # Draw bars
        bars = ax.bar(modes, y_vals, color=[color, "coral"], edgecolor="black", alpha=0.85, width=0.35)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.1)
        
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
            
    # 2. Plot latency comparison in the 6th slot
    ax = axes[1, 2]
    if "Latency_sec" in df.columns:
        latencies = pd.to_numeric(df["Latency_sec"], errors='coerce').fillna(0.0).tolist()
        bars = ax.bar(modes, latencies, color=["lightgrey", "brown"], edgecolor="black", alpha=0.85, width=0.35)
        ax.set_title("Inference Latency", fontsize=12, fontweight="semibold")
        ax.set_ylabel("Time (seconds)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        max_lat = max(latencies) if latencies else 1.0
        ax.set_ylim(0, max_lat * 1.25)
        
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, val + (max_lat * 0.02), f"{val:.2f}s", ha="center", va="bottom", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Latency not found", ha="center", va="center")
        ax.axis("off")
        
    plt.tight_layout()
    try:
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved TTA comparison plots to: {output_img_path}")
    except Exception as e:
        logger.error(f"Failed to save TTA comparison plots: {e}")
    finally:
        plt.close("all")


def aggregate_tta_comparison(
    baseline_metrics: Dict[str, float],
    tta_metrics: Dict[str, float],
    baseline_latency: float,
    tta_latency: float,
    output_dir: Path
) -> None:
    """Aggregates TTA inference metrics and generates comparison CSV and comparative bar charts."""
    records = []
    
    # 1. Compile baseline record
    records.append({
        "Mode": "Baseline",
        "Accuracy": baseline_metrics.get("accuracy", 0.0),
        "ROC_AUC": baseline_metrics.get("roc_auc", 0.0),
        "PR_AUC": baseline_metrics.get("pr_auc", 0.0),
        "Macro_F1": baseline_metrics.get("macro_f1", 0.0),
        "Balanced_Accuracy": baseline_metrics.get("balanced_accuracy", 0.0),
        "Sensitivity": baseline_metrics.get("sensitivity", 0.0),
        "Specificity": baseline_metrics.get("specificity", 0.0),
        "Latency_sec": baseline_latency
    })
    
    # 2. Compile TTA record
    records.append({
        "Mode": "TTA",
        "Accuracy": tta_metrics.get("accuracy", 0.0),
        "ROC_AUC": tta_metrics.get("roc_auc", 0.0),
        "PR_AUC": tta_metrics.get("pr_auc", 0.0),
        "Macro_F1": tta_metrics.get("macro_f1", 0.0),
        "Balanced_Accuracy": tta_metrics.get("balanced_accuracy", 0.0),
        "Sensitivity": tta_metrics.get("sensitivity", 0.0),
        "Specificity": tta_metrics.get("specificity", 0.0),
        "Latency_sec": tta_latency
    })
    
    csv_path = output_dir / "tta_comparison.csv"
    try:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved TTA comparison CSV table to: {csv_path}")
        
        plot_path = output_dir / "tta_comparison.png"
        generate_tta_comparison_plots(csv_path, plot_path)
    except Exception as e:
        logger.error(f"Failed to compile TTA comparison reports: {e}")
