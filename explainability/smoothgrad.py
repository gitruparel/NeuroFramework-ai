"""SmoothGrad saliency map smoothing."""

import torch
from core.interfaces import BaseModel
from explainability.base import MRIExplainer


class SmoothGrad(MRIExplainer):
    """Averages gradients across multiple noisy iterations of the input image."""

    def __init__(self, num_samples: int = 20, noise_level: float = 0.15):
        self.num_samples = num_samples
        self.noise_level = noise_level

    def generate_heatmap(self, model: BaseModel, tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        # Skeleton implementation
        return torch.zeros_like(tensor)
