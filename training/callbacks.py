"""Trainer callback hook definitions for early stopping, WandB logging, and history plotting."""

import json
from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
from core.interfaces import BaseCallback
from core.logging import setup_logger

logger = setup_logger("training.callbacks", "training/callbacks.log")


class EarlyStopping(BaseCallback):
    """Monitors metrics and terminates training loop early if performance plateaus."""

    def __init__(self, monitor: str = "val_loss", patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        if mode not in ("min", "max"):
            raise ValueError(f"EarlyStopping: Mode must be 'min' or 'max', got '{mode}'.")
            
        self.wait = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        score = metrics.get(self.monitor)
        if score is None:
            return

        improved = False
        if self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                improved = True
        elif self.mode == "max":
            if score > self.best_score + self.min_delta:
                self.best_score = score
                improved = True

        if improved:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                trainer.stop_training = True


class WandbLogger(BaseCallback):
    """Logs epoch metrics and execution configurations to Weights & Biases."""

    def __init__(self, project: str, config: Dict[str, Any] | None = None, **kwargs: Any):
        self.project = project
        self.config = config or {}
        self.kwargs = kwargs
        self.wandb = None
        self._active = False

    def on_train_start(self, trainer: Any) -> None:
        logger.info("Initializing WandB experiment logger...")
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=self.project, config=self.config, **self.kwargs)
            self._active = True
        except ImportError:
            logger.warning("WandbLogger: weights & biases package is not installed. Running in offline/console fallback mode.")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self._active and self.wandb is not None:
            self.wandb.log(metrics, step=epoch)
        else:
            logger.debug(f"Console log [Epoch {epoch + 1}]: {metrics}")

    def on_train_end(self, trainer: Any) -> None:
        if self._active and self.wandb is not None:
            self.wandb.finish()


class HistoryTracker(BaseCallback):
    """Tracks full epoch history, writing history.json and exporting metrics plots on finish."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list] = {}

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        self.history.setdefault("epoch", []).append(epoch)
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(v)

    def on_train_end(self, trainer: Any) -> None:
        # Save history.json
        history_path = self.output_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"Saved training history to {history_path}")

        # Plot loss and accuracy curves
        try:
            import matplotlib.pyplot as plt
            epochs = self.history.get("epoch", [])
            if not epochs:
                return

            # Plot loss history
            plt.figure()
            if "train_loss" in self.history:
                plt.plot(epochs, self.history["train_loss"], label="Train Loss")
            if "val_loss" in self.history:
                plt.plot(epochs, self.history["val_loss"], label="Val Loss")
            plt.title("Loss History")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend()
            loss_plot_path = self.output_dir / "loss.png"
            plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Plot accuracy history
            plt.figure()
            if "train_accuracy" in self.history:
                plt.plot(epochs, self.history["train_accuracy"], label="Train Accuracy")
            if "val_accuracy" in self.history:
                plt.plot(epochs, self.history["val_accuracy"], label="Val Accuracy")
            plt.title("Accuracy History")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.legend()
            acc_plot_path = self.output_dir / "accuracy.png"
            plt.savefig(acc_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved loss and accuracy visualization curves.")
        except Exception as e:
            logger.error(f"HistoryTracker failed to plot and save visualization curves: {e}")
