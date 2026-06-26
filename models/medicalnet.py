"""MedicalNet 3D ResNet backbone skeleton wrapper."""

import torch
from models.base_model import MRIModel


class MedicalNet3D(MRIModel):
    """3D ResNet model wrapper pre-trained on large medical datasets."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, shortcut_type: str = "B"):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.shortcut_type = shortcut_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skeleton forward return
        return x
