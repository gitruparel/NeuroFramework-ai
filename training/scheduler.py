"""Learning rate scheduler initialization factory."""

from typing import Any
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, ReduceLROnPlateau


def get_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs: Any) -> Any:
    """Helper factory to retrieve PyTorch LR Scheduler by name."""
    if name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, **kwargs)
    elif name == "StepLR":
        return StepLR(optimizer, **kwargs)
    elif name == "ExponentialLR":
        return ExponentialLR(optimizer, **kwargs)
    elif name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler name: {name}")
