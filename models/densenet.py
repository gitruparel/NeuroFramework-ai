"""3D DenseNet model wrapper utilizing MONAI's classifier architecture."""

import torch
from typing import Any
from monai.networks.nets import DenseNet121
from models.base_model import MRIModel


class DenseNet3D(MRIModel):
    """DenseNet 3D network classification model using MONAI's DenseNet121 implementation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, spatial_dims: int = 3, **kwargs: Any):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.spatial_dims = spatial_dims
        self.model = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding standard tensor input directly to MONAI backbone."""
        return self.model(x)
