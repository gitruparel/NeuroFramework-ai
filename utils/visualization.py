"""Visualization helper methods for plotting MRI slices and overlaying attributions."""

from typing import Any
import numpy as np


def plot_mri_slice(volume: np.ndarray, slice_idx: int, axis: int = 0) -> Any:
    """Extracts and formats single orthogonal slice from 3D volume for plotting."""
    if axis == 0:
        return volume[slice_idx, :, :]
    elif axis == 1:
        return volume[:, slice_idx, :]
    else:
        return volume[:, :, slice_idx]


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays generated explainability heatmap on top of target brain scan slice."""
    # Skeleton utility returning blank overlay
    return image
