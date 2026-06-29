"""Trainer checkpoint manager saving best and latest model states."""

from pathlib import Path
from typing import Any, Dict
import torch
from core.interfaces import BaseCallback
from core.logging import setup_logger

logger = setup_logger("training.checkpoint", "training/checkpoint.log")


class Checkpointer(BaseCallback):
    """Saves and restores model weights, optimizer states, and history parameters."""

    def __init__(self, dirpath: str | Path, monitor: str = "val_loss", mode: str = "min"):
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        
        if mode not in ("min", "max"):
            raise ValueError(f"Checkpointer: Mode must be 'min' or 'max', got '{mode}'.")
            
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        val = metrics.get(self.monitor)
        if val is None:
            logger.warning(f"Checkpointer: Monitored metric '{self.monitor}' was not found in metrics dictionary.")
            return

        is_best = False
        if self.mode == "min":
            if val < self.best_metric:
                self.best_metric = val
                is_best = True
        elif self.mode == "max":
            if val > self.best_metric:
                self.best_metric = val
                is_best = True

        # Pack checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                trainer.model.module.state_dict()
                if isinstance(trainer.model, torch.nn.DataParallel)
                else trainer.model.state_dict()
            ),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict()
                if getattr(trainer, "scheduler", None) is not None
                else None
            ),
            "best_metric": self.best_metric,
            "monitor": self.monitor,
            "mode": self.mode,
            "metrics": metrics,
            "config": trainer.config,
        }

        # 1. Always save the latest model for training resumption
        latest_path = self.dirpath / "latest_model.pt"
        torch.save(checkpoint, latest_path)
        logger.debug(f"Saved latest checkpoint at {latest_path}")

        # 2. Save best model if monitored metric improved
        if is_best:
            best_path = self.dirpath / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                f"New best metric '{self.monitor}'={val:.4f} achieved. Saved best checkpoint to {best_path}"
            )
