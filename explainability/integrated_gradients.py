"""Integrated Gradients attribution calculation."""

import torch
from core.interfaces import BaseModel
from explainability.base import MRIExplainer


class IntegratedGradients(MRIExplainer):
    """Calculates attribution by integrating gradients along path from baseline to input."""

    def __init__(self, steps: int = 50):
        self.steps = steps

    def generate_heatmap(self, model: BaseModel, tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        # Skeleton implementation
        return torch.zeros_like(tensor)
