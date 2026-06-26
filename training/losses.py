"""Loss functions definitions for structural MRI analysis models."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Computes soft Dice loss coefficient for 3D/2D segmentation masks."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
