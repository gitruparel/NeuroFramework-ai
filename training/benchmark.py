"""Model summary and benchmark visualization utilities."""

import json
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from core.logging import setup_logger

logger = setup_logger("training.benchmark", "training/benchmark.log")


def generate_model_summary(model: nn.Module, input_shape: Tuple[int, ...], output_path: Path) -> None:
    """Generates an execution layer details summary and writes it to a text file.
    
    Computes module parameters, layer output tensor dimensions, and estimates
    activation memory footprint using forward hooks.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Model size estimation: 4 bytes per float32 parameter
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"Model Summary: {model.__class__.__name__}")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Input Shape: {input_shape}")
    summary_lines.append(f"Total Parameters: {total_params:,}")
    summary_lines.append(f"Trainable Parameters: {trainable_params:,}")
    summary_lines.append(f"Non-Trainable Parameters: {non_trainable_params:,}")
    summary_lines.append(f"Estimated Parameter Size: {model_size_mb:.2f} MB")
    
    activation_sizes: List[int] = []
    hooks = []
    
    def hook_fn(module, input_t, output_t):
        if isinstance(output_t, torch.Tensor):
            activation_sizes.append(output_t.numel())
        elif isinstance(output_t, (list, tuple)):
            for o in output_t:
                if isinstance(o, torch.Tensor):
                    activation_sizes.append(o.numel())
                    
    # Register hooks on leaf modules to trace output shapes
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))
            
    # Run a forward pass on CPU to gather layer activations sizes
    try:
        # Move model to CPU temporarily to avoid cuda memory allocation errors
        original_device = next(model.parameters()).device
        model.cpu()
        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            model(dummy_input)
        model.to(original_device)
        
        activation_mem_mb = (sum(activation_sizes) * 4) / (1024 * 1024)
        summary_lines.append(f"Estimated Activation Memory Footprint: {activation_mem_mb:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not run forward pass for model summary activation trace: {e}")
        activation_mem_mb = 0.0
        summary_lines.append("Estimated Activation Memory Footprint: N/A (Forward pass failed)")
    finally:
        for h in hooks:
            h.remove()
            
    summary_lines.append("=" * 80)
    summary_lines.append(f"{'Layer Name':<45} {'Layer Type':<20} {'Parameter Count':<15}")
    summary_lines.append("-" * 80)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            layer_params = sum(p.numel() for p in module.parameters())
            summary_lines.append(f"{name[:44]:<45} {module.__class__.__name__[:19]:<20} {layer_params:<15,}")
            
    summary_lines.append("=" * 80)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        logger.info(f"Model summary successfully written to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write model_summary.txt: {e}")


def generate_benchmark_plots(csv_path: Path, output_img_path: Path) -> None:
    """Generates a comparison grid bar plot of accuracy, AUC, and parameters for all models."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if not csv_path.exists():
        logger.error(f"Cannot generate benchmark plots. CSV file not found: {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read benchmark CSV: {e}")
        return
        
    plt.close("all")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Architecture Benchmark Comparison", fontsize=16, fontweight="bold")
    
    metrics = [
        ("Val_Accuracy", "Validation Accuracy", "royalblue"),
        ("ROC_AUC", "ROC-AUC Score", "seagreen"),
        ("PR_AUC", "PR-AUC Score", "orange"),
        ("F1", "Macro F1-Score", "purple"),
        ("Training_Time", "Training Time (seconds)", "crimson"),
        ("Parameter_Count", "Parameter Count (Millions)", "darkgrey")
    ]
    
    for idx, (col, title, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Metric '{col}' not found", ha="center", va="center")
            ax.axis("off")
            continue
            
        x_vals = df["Architecture"]
        y_vals = df[col]
        
        # Clean potential non-numeric strings
        y_vals = pd.to_numeric(y_vals, errors='coerce').fillna(0.0)
        
        # If parameter count, convert to millions for clean plotting
        if col == "Parameter_Count":
            y_vals = y_vals / 1e6
            
        ax.bar(x_vals, y_vals, color=color, edgecolor="black", alpha=0.85, width=0.4)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_ylabel(title if col == "Parameter_Count" else col)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Add value labels above bars
        for i, val in enumerate(y_vals):
            if col == "Parameter_Count" or col == "Training_Time":
                ax.text(i, val + (val * 0.02) + 1e-5, f"{val:,.2f}", ha="center", va="bottom", fontsize=10)
            else:
                ax.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
                ax.set_ylim(0, 1.1)
                
    plt.tight_layout()
    try:
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved benchmark comparison plots to: {output_img_path}")
    except Exception as e:
        logger.error(f"Failed to save benchmark comparison plots: {e}")
    finally:
        plt.close("all")


def aggregate_architecture_benchmark(experiment_dir: Path, architectures: list) -> None:
    """Aggregates metadata from each architecture subdirectory and saves a ranked comparison CSV and PNG."""
    records = []
    for arch in architectures:
        meta_path = experiment_dir / arch / "experiment_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    
                # Map metadata fields to comparison columns
                record = {
                    "Architecture": arch,
                    "Val_Loss": meta.get("best_val_loss", 0.0),
                    "Val_Accuracy": meta.get("best_val_accuracy", 0.0),
                    "ROC_AUC": meta.get("best_val_roc_auc", 0.0),
                    "PR_AUC": meta.get("best_val_pr_auc", 0.0),
                    "F1": meta.get("best_val_macro_f1", meta.get("macro_f1", 0.0)),
                    "Balanced_Accuracy": meta.get("best_val_balanced_accuracy", 0.0),
                    "Training_Time": meta.get("training_time", 0.0),
                    "Epoch_Time": meta.get("epoch_time", 0.0),
                    "Parameter_Count": meta.get("parameter_count", 0),
                    "Model_Size": meta.get("model_size_mb", 0.0),
                    "Best_Epoch": meta.get("best_epoch", 0),
                    "Learning_Rate": meta.get("lr", 0.0),
                    "Optimizer": meta.get("optimizer", ""),
                    "Scheduler": meta.get("scheduler", ""),
                    "Augmentation_Profile": meta.get("augmentation_profile", "none")
                }
                records.append(record)
            except Exception as e:
                logger.error(f"Failed to read metadata for architecture {arch} during aggregation: {e}")
                
    if not records:
        logger.warning("No experiment metadata found to aggregate.")
        return
        
    import pandas as pd
    csv_path = experiment_dir / "architecture_comparison.csv"
    try:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved architecture comparison table to: {csv_path}")
        
        plot_path = experiment_dir / "architecture_comparison.png"
        generate_benchmark_plots(csv_path, plot_path)
    except Exception as e:
        logger.error(f"Failed to save architecture comparison CSV: {e}")
