"""Experiment configurations, reproducibility seeding, and run directories management."""

import random
from pathlib import Path
import numpy as np
import torch
from core.logging import setup_logger

logger = setup_logger("training.experiment", "training/experiment.log")


def set_seed(seed: int = 42) -> None:
    """Sets system-wide seed for reproducibility across random, numpy, torch, and MONAI."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import monai
        monai.utils.set_determinism(seed=seed)
    except Exception as e:
        logger.warning(f"Could not set MONAI determinism: {e}")
    logger.info(f"Reproducibility seed set to {seed}")


def create_experiment_run(experiments_dir: str | Path, exp_name: str) -> Path:
    """Creates directory for saving configs, weights, and logs for a unique experiment run."""
    run_path = Path(experiments_dir) / exp_name
    run_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created experiment run directory at: {run_path}")
    return run_path
