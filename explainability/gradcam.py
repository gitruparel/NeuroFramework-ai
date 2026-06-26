"""Grad-CAM attribution map generation."""

import torch
from core.interfaces import BaseModel
from explainability.base import MRIExplainer


class GradCAM(MRIExplainer):
    """Computes Grad-CAM saliency maps for target layers in 3D networks."""

    def __init__(self, target_layer_name: str):
        self.target_layer_name = target_layer_name

    def generate_heatmap(self, model: BaseModel, tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        # Skeleton implementation
        return torch.zeros_like(tensor)
