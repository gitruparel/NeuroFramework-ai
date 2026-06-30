"""Transfer learning benchmarking and comparison utilities."""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from core.logging import setup_logger

logger = setup_logger("training.transfer_learning", "training/transfer_learning.log")


def generate_transfer_learning_plots(csv_path: Path, output_img_path: Path) -> None:
    """Generates comparison bar plots comparing metrics across transfer learning initialization types."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if not csv_path.exists():
        logger.error(f"Cannot generate transfer learning comparison plots. CSV not found: {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read transfer learning comparison CSV: {e}")
        return
        
    plt.close("all")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Transfer Learning Initialization Benchmark Comparison", fontsize=16, fontweight="bold")
    
    metrics = [
        ("Val_Accuracy", "Validation Accuracy", "royalblue", 1.1),
        ("ROC_AUC", "ROC-AUC Score", "seagreen", 1.1),
        ("PR_AUC", "PR-AUC Score", "orange", 1.1),
        ("Training_Time", "Training Duration (sec)", "purple", None),
        ("Epoch_to_Best_Validation", "Convergence Speed (Best Epoch)", "crimson", None)
    ]
    
    # 1. Plot standard metrics
    for idx, (col, title, color, ylim) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Metric '{col}' not found", ha="center", va="center")
            ax.axis("off")
            continue
            
        x_vals = df["Initialization_Type"]
        y_vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        ax.bar(x_vals, y_vals, color=color, edgecolor="black", alpha=0.85, width=0.4)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        if ylim:
            ax.set_ylim(0, ylim)
        else:
            max_y = max(y_vals) if len(y_vals) > 0 else 1.0
            ax.set_ylim(0, max_y * 1.25 if max_y > 0 else 1.0)
            
        for i, val in enumerate(y_vals):
            label_fmt = f"{val:.4f}" if col in ["Val_Accuracy", "ROC_AUC", "PR_AUC"] else (f"{val:.1f}s" if col == "Training_Time" else f"Ep {int(val)}")
            ax.text(i, val + (ax.get_ylim()[1] * 0.01), label_fmt, ha="center", va="bottom", fontsize=10)
            
    # Leave 6th slot empty or show a summary note
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(
        0.5, 0.5, 
        "Comparison Summary:\n\nEvaluating the influence\nof Pretrained weight\ninitialization strategies\non 3D CNN backbones.", 
        ha="center", va="center", fontsize=12, style="italic", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    plt.tight_layout()
    try:
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved transfer learning comparison plots to: {output_img_path}")
    except Exception as e:
        logger.error(f"Failed to save transfer learning comparison plots: {e}")
    finally:
        plt.close("all")


def aggregate_transfer_learning_benchmark(experiment_dir: Path, init_types: List[str]) -> None:
    """Aggregates metadata across initialization runs, producing transfer_learning_comparison.csv and plots."""
    records = []
    for init in init_types:
        meta_path = experiment_dir / init / "experiment_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    
                record = {
                    "Initialization_Type": init,
                    "Val_Accuracy": meta.get("best_val_accuracy", 0.0),
                    "ROC_AUC": meta.get("best_val_roc_auc", 0.0),
                    "PR_AUC": meta.get("best_val_pr_auc", 0.0),
                    "Macro_F1": meta.get("best_val_macro_f1", 0.0),
                    "Balanced_Accuracy": meta.get("best_val_balanced_accuracy", 0.0),
                    "Epoch_to_Best_Validation": meta.get("best_epoch", 0),
                    "Training_Time": meta.get("training_time", 0.0),
                    "Parameter_Count": meta.get("parameter_count", 0),
                }
                records.append(record)
            except Exception as e:
                logger.error(f"Failed to read metadata for initialization {init} during aggregation: {e}")
                
    if not records:
        logger.warning("No experiment metadata found to aggregate for transfer learning.")
        return
        
    import pandas as pd
    csv_path = experiment_dir / "transfer_learning_comparison.csv"
    try:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved transfer learning comparison table to: {csv_path}")
        
        plot_path = experiment_dir / "transfer_learning_comparison.png"
        generate_transfer_learning_plots(csv_path, plot_path)
    except Exception as e:
        logger.error(f"Failed to save transfer learning comparison CSV: {e}")
