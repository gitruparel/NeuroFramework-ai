"""Configurable medical image augmentation framework using MONAI 3D transforms."""

from pathlib import Path
from typing import Dict, Any, Callable
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandAffine,
    RandZoom,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    RandShiftIntensity,
    RandScaleIntensity,
    Rand3DElastic
)


def get_mri_augmentations(profile: str, seed: int = 42) -> Compose:
    """Builds a Compose pipeline of MONAI random 3D augmentations based on the profile name.
    
    Ensures complete reproducibility by using set_determinism and fixed parameters.
    """
    # 1. Spatial & Intensity Augmentations Configuration per Profile
    transforms = []
    
    if profile == "minimal":
        # - RandomFlip (spatial_axis=0 is Left-Right flip in RAS orientation)
        transforms.append(RandFlip(prob=0.5, spatial_axis=0))
        # - RandShiftIntensity
        transforms.append(RandShiftIntensity(prob=0.5, offsets=(-0.08, 0.08)))
        
    elif profile == "moderate":
        # - RandomFlip
        transforms.append(RandFlip(prob=0.5, spatial_axis=0))
        # - RandAffine (±10° rotation, translation, scaling 0.9-1.1)
        transforms.append(RandAffine(
            prob=0.5,
            rotate_range=(np.pi/18, np.pi/18, np.pi/18),  # ±10 degrees
            translate_range=(4, 4, 4),  # ±4 voxels translation
            scale_range=(0.1, 0.1, 0.1),  # 1.0 ± 0.1 scaling
            padding_mode="zeros"
        ))
        # - RandAdjustContrast
        transforms.append(RandAdjustContrast(prob=0.5, gamma=(0.5, 2.0)))
        # - RandGaussianNoise
        transforms.append(RandGaussianNoise(prob=0.5, mean=0.0, std=0.015))
        
    elif profile == "strong":
        # - RandomFlip
        transforms.append(RandFlip(prob=0.5, spatial_axis=0))
        # - RandRotate90
        transforms.append(RandRotate90(prob=0.5, max_k=3, spatial_axes=(1, 2)))
        # - RandAffine
        transforms.append(RandAffine(
            prob=0.5,
            rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            translate_range=(4, 4, 4),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="zeros"
        ))
        # - RandZoom
        transforms.append(RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1, padding_mode="constant", constant_values=0))
        # - RandGaussianNoise
        transforms.append(RandGaussianNoise(prob=0.5, mean=0.0, std=0.015))
        # - RandGaussianSmooth
        transforms.append(RandGaussianSmooth(prob=0.5, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)))
        # - RandAdjustContrast
        transforms.append(RandAdjustContrast(prob=0.5, gamma=(0.5, 2.0)))
        # - RandShiftIntensity
        transforms.append(RandShiftIntensity(prob=0.5, offsets=(-0.08, 0.08)))
        # - RandScaleIntensity
        transforms.append(RandScaleIntensity(prob=0.5, factors=(-0.1, 0.1)))
        
    elif profile == "research":
        # - RandomFlip
        transforms.append(RandFlip(prob=0.5, spatial_axis=0))
        # - RandRotate90
        transforms.append(RandRotate90(prob=0.5, max_k=3, spatial_axes=(1, 2)))
        # - RandAffine
        transforms.append(RandAffine(
            prob=0.5,
            rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            translate_range=(4, 4, 4),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="zeros"
        ))
        # - RandZoom
        transforms.append(RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1, padding_mode="constant", constant_values=0))
        # - Rand3DElastic (conservative parameters for realistic brain MRI structures)
        transforms.append(Rand3DElastic(
            prob=0.15,
            sigma_range=(5, 8),
            magnitude_range=(50, 100),
            padding_mode="zeros"
        ))
        # - RandGaussianNoise
        transforms.append(RandGaussianNoise(prob=0.5, mean=0.0, std=0.015))
        # - RandGaussianSmooth
        transforms.append(RandGaussianSmooth(prob=0.5, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)))
        # - RandAdjustContrast
        transforms.append(RandAdjustContrast(prob=0.5, gamma=(0.5, 2.0)))
        # - RandShiftIntensity
        transforms.append(RandShiftIntensity(prob=0.5, offsets=(-0.08, 0.08)))
        # - RandScaleIntensity
        transforms.append(RandScaleIntensity(prob=0.5, factors=(-0.1, 0.1)))
        
    else:
        raise ValueError(f"Unknown augmentation profile: {profile}")
        
    # Return pipeline wrapped in monai Compose
    return Compose(transforms)


def get_profile_metadata(profile: str) -> Dict[str, Any]:
    """Retrieves metadata describing enabled transforms and their probabilities."""
    if profile == "minimal":
        return {
            "augmentation_profile": "minimal",
            "enabled_transforms": ["RandFlip", "RandShiftIntensity"],
            "transform_probabilities": {
                "RandFlip": 0.5,
                "RandShiftIntensity": 0.5
            }
        }
    elif profile == "moderate":
        return {
            "augmentation_profile": "moderate",
            "enabled_transforms": ["RandFlip", "RandAffine", "RandAdjustContrast", "RandGaussianNoise"],
            "transform_probabilities": {
                "RandFlip": 0.5,
                "RandAffine": 0.5,
                "RandAdjustContrast": 0.5,
                "RandGaussianNoise": 0.5
            }
        }
    elif profile == "strong":
        return {
            "augmentation_profile": "strong",
            "enabled_transforms": [
                "RandFlip", "RandRotate90", "RandAffine", "RandZoom",
                "RandGaussianNoise", "RandGaussianSmooth", "RandAdjustContrast",
                "RandShiftIntensity", "RandScaleIntensity"
            ],
            "transform_probabilities": {
                "RandFlip": 0.5,
                "RandRotate90": 0.5,
                "RandAffine": 0.5,
                "RandZoom": 0.5,
                "RandGaussianNoise": 0.5,
                "RandGaussianSmooth": 0.5,
                "RandAdjustContrast": 0.5,
                "RandShiftIntensity": 0.5,
                "RandScaleIntensity": 0.5
            }
        }
    elif profile == "research":
        return {
            "augmentation_profile": "research",
            "enabled_transforms": [
                "RandFlip", "RandRotate90", "RandAffine", "RandZoom", "Rand3DElastic",
                "RandGaussianNoise", "RandGaussianSmooth", "RandAdjustContrast",
                "RandShiftIntensity", "RandScaleIntensity"
            ],
            "transform_probabilities": {
                "RandFlip": 0.5,
                "RandRotate90": 0.5,
                "RandAffine": 0.5,
                "RandZoom": 0.5,
                "Rand3DElastic": 0.15,
                "RandGaussianNoise": 0.5,
                "RandGaussianSmooth": 0.5,
                "RandAdjustContrast": 0.5,
                "RandShiftIntensity": 0.5,
                "RandScaleIntensity": 0.5
            }
        }
    else:
        raise ValueError(f"Unknown augmentation profile: {profile}")


def generate_augmentation_preview(tensor: torch.Tensor, augment_fn: Callable[[torch.Tensor], torch.Tensor], output_path: Path) -> None:
    """Generates and saves a slice validation preview showing original vs 5 augmented variations."""
    # Ensure input shape matches (1, D, H, W)
    if len(tensor.shape) != 4:
        raise ValueError(f"Expected 3D volume tensor with channels shape (1, D, H, W), got {tensor.shape}")
        
    d_dim = tensor.shape[1]
    mid_idx = d_dim // 2
    
    # Slices will be extracted along the D axis (axial/sagittal plane slice)
    original_slice = tensor[0, mid_idx, :, :]
    
    plt.close("all")
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    
    # 1. Plot original
    axes[0].imshow(original_slice.cpu().numpy(), cmap="gray", origin="lower")
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    # 2. Plot 5 random augmentations
    for i in range(5):
        aug_tensor = augment_fn(tensor)
        aug_slice = aug_tensor[0, mid_idx, :, :]
        axes[i + 1].imshow(aug_slice.cpu().numpy(), cmap="gray", origin="lower")
        axes[i + 1].set_title(f"Augment {i + 1}")
        axes[i + 1].axis("off")
        
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def generate_augmentation_plots(csv_path: Path, output_img_path: Path) -> None:
    """Generates comparison bar plots ranking performance metrics across augmentation profiles."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if not csv_path.exists():
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
        
    plt.close("all")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Augmentation Profile Benchmark Comparison", fontsize=16, fontweight="bold")
    
    metrics = [
        ("Val_Accuracy", "Validation Accuracy", "royalblue"),
        ("ROC_AUC", "ROC-AUC Score", "seagreen"),
        ("PR_AUC", "PR-AUC Score", "orange"),
        ("F1", "Macro F1-Score", "purple"),
        ("Balanced_Accuracy", "Balanced Accuracy", "crimson"),
        ("Training_Time", "Training Time (seconds)", "darkgrey")
    ]
    
    for idx, (col, title, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Metric '{col}' not found", ha="center", va="center")
            ax.axis("off")
            continue
            
        x_vals = df["Augmentation_Profile"]
        y_vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        ax.bar(x_vals, y_vals, color=color, edgecolor="black", alpha=0.85, width=0.4)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        if col != "Training_Time":
            ax.set_ylim(0, 1.1)
            
        for i, val in enumerate(y_vals):
            if col == "Training_Time":
                ax.text(i, val + (val * 0.02) + 1e-5, f"{val:,.1f}", ha="center", va="bottom", fontsize=10)
            else:
                ax.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
                
    plt.tight_layout()
    try:
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    except Exception:
        pass
    finally:
        plt.close("all")


def aggregate_augmentations_benchmark(experiment_dir: Path, profiles: list) -> None:
    """Aggregates metadata across augmentation experiment folders, compiling CSV and plots."""
    import json
    records = []
    for profile in profiles:
        meta_path = experiment_dir / profile / "experiment_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    
                record = {
                    "Augmentation_Profile": profile,
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
            except Exception:
                pass
                
    if not records:
        return
        
    import pandas as pd
    csv_path = experiment_dir / "augmentation_comparison.csv"
    try:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        
        plot_path = experiment_dir / "augmentation_comparison.png"
        generate_augmentation_plots(csv_path, plot_path)
    except Exception:
        pass
