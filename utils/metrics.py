"""Medical image validation metrics implementation."""

import numpy as np


def calculate_dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes standard binary Dice similarity coefficient."""
    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)
