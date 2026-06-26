"""Optimizer registration and building helper utilities."""

from typing import Any, Dict, Iterator
import torch
from torch.optim import AdamW, SGD


def get_optimizer(name: str, params: Iterator[torch.nn.Parameter], lr: float, **kwargs: Any) -> torch.optim.Optimizer:
    """Helper factory to instantiate optimizers by name."""
    if name.lower() == "adamw":
        return AdamW(params, lr=lr, **kwargs)
    elif name.lower() == "sgd":
        return SGD(params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer name: {name}")
