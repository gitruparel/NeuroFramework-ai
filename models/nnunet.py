"""nnU-Net segmentation backbone wrapper."""

import torch
from models.base_model import MRIModel


class NNUnet3D(MRIModel):
    """3D U-Net configured dynamically using nnU-Net heuristic guidelines."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skeleton forward return
        return x
