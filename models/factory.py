"""Centralized model constructor factory for 3D MRI classifiers."""

from typing import Any
from models.base_model import MRIModel
from models.densenet import DenseNet3D
from models.resnet import ResNet3D


class ModelFactory:
    """Factory responsible for constructing MRI classifier architectures."""
    
    @staticmethod
    def create_model(
        model_name: str,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float = 0.0,
        **kwargs: Any
    ) -> MRIModel:
        """Constructs a registered model based on the architecture identifier."""
        name_lower = model_name.lower().replace("-", "").replace("_", "")
        if name_lower == "densenet121":
            return DenseNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_prob=dropout_prob,
                **kwargs
            )
        elif name_lower == "resnet10":
            return ResNet3D(
                depth=10,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_prob=dropout_prob,
                **kwargs
            )
        elif name_lower == "resnet18":
            return ResNet3D(
                depth=18,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_prob=dropout_prob,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown architecture model name: {model_name}")
