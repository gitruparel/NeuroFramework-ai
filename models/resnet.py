"""3D ResNet model wrapper utilizing MONAI's ResNet implementations."""

from typing import Any
import torch
import torch.nn as nn
import monai.networks.nets as nets
from models.base_model import MRIModel


class ResNet3D(MRIModel):
    """ResNet 3D classification network (ResNet-10 or ResNet-18) using MONAI backbones."""

    def __init__(self, depth: int, in_channels: int = 1, out_channels: int = 2, dropout_prob: float = 0.0, **kwargs: Any):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.depth = depth
        self.dropout_prob = dropout_prob

        # Initialize MONAI ResNet backbone without standard feed-forward classification head
        if depth == 10:
            self.backbone = nets.resnet10(
                spatial_dims=3,
                n_input_channels=in_channels,
                feed_forward=False,
                **kwargs
            )
        elif depth == 18:
            self.backbone = nets.resnet18(
                spatial_dims=3,
                n_input_channels=in_channels,
                feed_forward=False,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        # The feature representation dimension of ResNet10/18 features layer is 512
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_prob) if dropout_prob > 0.0 else nn.Identity(),
            nn.Linear(512, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting backbone representations and running classifier head."""
        features = self.backbone(x)
        return self.head(features)
