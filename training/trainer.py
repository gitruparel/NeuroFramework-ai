"""Training execution loop implementation coordinating PyTorch modules, optimization steps, and callbacks."""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from schemas.dataset import collate_dataset_samples
from core.interfaces import BaseTrainer, BaseCallback, BaseModel, BaseDataset
from core.logging import setup_logger
from training.metrics import MetricsManager
from tqdm import tqdm
from utils.device import resolve_backend

logger = setup_logger("training.trainer", "training/trainer.log")


class Trainer(BaseTrainer):
    """Training engine managing dataloaders, optimization loops, AMP, grad clipping, and lifecycles."""

    def __init__(
        self,
        model: BaseModel,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
        optimizer: torch.optim.Optimizer,
        loss_fn: Any,
        scheduler: Any = None,
        callbacks: List[BaseCallback] | None = None,
        config: Dict[str, Any] | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.config = config or {}
        
        self.backend = resolve_backend(self.config.get("device", "auto"))
        self.device = self.backend.device
        self.current_epoch = 0
        self.start_epoch = 0
        self.stop_training = False

    def fit(self, resume_from: str | Path | None = None) -> None:
        """Loads weight checkpoints and optimizers state if provided, then executes the train loop."""
        # Move model to target device BEFORE loading optimizer state dict so PyTorch aligns state devices to model parameters
        self.model.to(self.device)
        
        if resume_from is not None:
            checkpoint_path = Path(resume_from)
            if checkpoint_path.exists():
                logger.info(f"Trainer: Resuming training run from checkpoint: {checkpoint_path}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                except Exception as e:
                    logger.error(
                        f"Trainer: Failed to load checkpoint {checkpoint_path} using map_location '{self.device}'. "
                        f"This might be due to PyTorch version incompatibilities or device serialization mismatches "
                        f"between platforms (e.g. loading a DirectML checkpoint on CPU/CUDA).\n"
                        f"Detailed error: {e}"
                    )
                    raise e
                
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
                # Move optimizer state tensors explicitly to target device to prevent any CPU/GPU mismatch
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                
                if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    
                self.start_epoch = checkpoint["epoch"] + 1
                self.current_epoch = self.start_epoch
                logger.info(f"Trainer: Successfully resumed. Next epoch will start at {self.start_epoch + 1}.")
            else:
                logger.warning(
                    f"Trainer: Resuming path {checkpoint_path} not found. Starting from scratch."
                )
                self.start_epoch = 0
                self.current_epoch = 0
        else:
            self.start_epoch = 0
            self.current_epoch = 0

        self.train()

    def train(self) -> None:
        """Starts model training execution loop."""
        logger.info(f"Trainer: Initializing training run on device '{self.device}'...")
        self.model.to(self.device)
        self.stop_training = False

        if self.backend.capabilities.benchmark:
            torch.backends.cudnn.benchmark = True

        batch_size = self.config.get("batch_size", 4)
        num_workers = self.config.get("num_workers", 2 if self.backend.capabilities.pin_memory else 0)
        pin_memory = self.config.get("pin_memory", self.backend.capabilities.pin_memory)
        persistent_workers = self.config.get("persistent_workers", num_workers > 0) if num_workers > 0 else False

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_dataset_samples
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_dataset_samples
        )

        # Set up Mixed Precision Scaler
        use_amp = self.config.get("amp", False) and self.backend.capabilities.amp
        device_type = "cuda" if self.backend.capabilities.amp else "cpu"
        scaler = torch.amp.GradScaler(device=device_type, enabled=use_amp)

        epochs = self.config.get("epochs", 10)
        start_epoch = getattr(self, "start_epoch", 0)

        # 1. Trigger train start callbacks
        for callback in self.callbacks:
            callback.on_train_start(self)

        for epoch in range(start_epoch, epochs):
            if self.stop_training:
                logger.info("Trainer: stop_training flag is active. Aborting epoch loop.")
                break

            self.current_epoch = epoch
            logger.info(f"Trainer: Epoch {epoch + 1}/{epochs}")

            # Training epoch
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            train_probs = []

            # Batch-level progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=True)
            for batch in pbar:
                inputs = batch["image"].to(self.device, non_blocking=self.backend.capabilities.non_blocking)
                targets = batch["label"].to(self.device, non_blocking=self.backend.capabilities.non_blocking)

                self.optimizer.zero_grad()

                # AMP Forward pass
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient Clipping
                grad_clip = self.config.get("grad_clip_max_norm")
                if grad_clip is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss.item() * inputs.size(0)

                # Track predictions and targets
                probs = torch.softmax(outputs, dim=1) if outputs.size(1) > 1 else torch.sigmoid(outputs)
                preds = torch.argmax(outputs, dim=1) if outputs.size(1) > 1 else (outputs > 0).long()

                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
                train_probs.extend(probs.detach().cpu().numpy())

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Compile Train Metrics
            train_loss = train_loss / len(self.train_dataset)
            train_metrics = MetricsManager.calculate_classification_metrics(
                np.array(train_targets),
                np.array(train_preds),
                np.array(train_probs)
            )

            # Validation epoch
            val_metrics = self.validate(val_loader=val_loader)

            # Step learning rate scheduler if present
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Get current learning rate from optimizer
            current_lr = 0.0
            for param_group in self.optimizer.param_groups:
                current_lr = param_group["lr"]
                break

            # Consolidate epoch metrics
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "lr": current_lr
            }
            for k, v in train_metrics.items():
                epoch_metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                if k != "val_loss":
                    epoch_metrics[f"val_{k}"] = v

            # 2. Trigger epoch end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, epoch_metrics)

        # 3. Trigger train end callbacks
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(self)

        logger.info("Trainer: Training run finished.")

    def validate(self, val_loader: DataLoader | None = None) -> Dict[str, float]:
        """Runs validation loop and returns calculated metrics."""
        use_amp = self.config.get("amp", False) and self.backend.capabilities.amp
        device_type = "cuda" if self.backend.capabilities.amp else "cpu"

        if val_loader is None:
            batch_size = self.config.get("batch_size", 4)
            num_workers = self.config.get("num_workers", 2 if self.backend.capabilities.pin_memory else 0)
            pin_memory = self.config.get("pin_memory", self.backend.capabilities.pin_memory)
            persistent_workers = self.config.get("persistent_workers", num_workers > 0) if num_workers > 0 else False

            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=collate_dataset_samples
            )

        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.get('epochs', 10)} [Val]", leave=False)
            for batch in pbar:
                inputs = batch["image"].to(self.device, non_blocking=self.backend.capabilities.non_blocking)
                targets = batch["label"].to(self.device, non_blocking=self.backend.capabilities.non_blocking)

                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1) if outputs.size(1) > 1 else torch.sigmoid(outputs)
                preds = torch.argmax(outputs, dim=1) if outputs.size(1) > 1 else (outputs > 0).long()

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss = val_loss / len(self.val_dataset)
        metrics = MetricsManager.calculate_classification_metrics(
            np.array(val_targets),
            np.array(val_preds),
            np.array(val_probs)
        )

        # Assemble validation outputs
        val_outputs = {"val_loss": val_loss}
        for k, v in metrics.items():
            val_outputs[f"val_{k}"] = v

        return val_outputs
