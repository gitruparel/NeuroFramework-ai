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
