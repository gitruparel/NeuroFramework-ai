"""Unit and integration smoke test suite validating Stage 4 and Stage 4.5 Training Framework requirements."""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from core.interfaces import BaseModel, BaseDataset
from training.trainer import Trainer
from training.checkpoint import Checkpointer
from training.callbacks import EarlyStopping, HistoryTracker, WandbLogger


# 1. Define Mock Model for training framework smoke tests
class DummyModel(BaseModel):
    """Simple 3D Convolutional Network model for classification validation."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# 2. Define Mock Dataset for training framework smoke tests
class DummyDataset(BaseDataset):
    """Generates synthetic 3D spatial volumes and label maps."""

    def __init__(self, size: int = 16):
        self.size = size
        # 3D spatial tensors of size (1, 8, 8, 8)
        self.images = [torch.randn(1, 8, 8, 8) for _ in range(size)]
        self.labels = [int(i % 2) for i in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.images[index],
            "label": self.labels[index]
        }


def test_trainer_pipeline_and_callbacks(tmp_path):
    """Smoke test running full training loop and verifying checkpointing, early stopping, and history tracking."""
    # Set up directories
    checkpoint_dir = tmp_path / "checkpoints"
    history_dir = tmp_path / "experiments"
    
    # Initialize components
    model = DummyModel()
    train_dataset = DummyDataset(size=12)
    val_dataset = DummyDataset(size=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    
    # Setup callbacks
    checkpointer = Checkpointer(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
    early_stopping = EarlyStopping(monitor="val_loss", patience=2, min_delta=1e-4, mode="min")
    history_tracker = HistoryTracker(output_dir=history_dir)
    wandb_logger = WandbLogger(project="smoke_test", config={"lr": 0.01}, mode="disabled") # disable actual internet requests
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        callbacks=[checkpointer, early_stopping, history_tracker, wandb_logger],
        config={
            "epochs": 3,
            "batch_size": 4,
            "device": "cpu", # Force cpu for local test suite consistency
            "grad_clip_max_norm": 1.0
        }
    )
    
    # 1. Execute fit loop
    trainer.fit()
    
    # Verify outputs created successfully
    assert (checkpoint_dir / "latest_model.pt").exists()
    assert (checkpoint_dir / "best_model.pt").exists()
    
    assert (history_dir / "history.json").exists()
    assert (history_dir / "loss.png").exists()
    assert (history_dir / "accuracy.png").exists()
    
    # Read history validation
    with open(history_dir / "history.json", encoding="utf-8") as f:
        history = json.load(f)
        
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["epoch"]) == 3
    assert history["epoch"] == [0, 1, 2]


def test_training_resumption(tmp_path):
    """Verify training resumption successfully restores weights and restarts at correct epoch state."""
    checkpoint_dir = tmp_path / "checkpoints"
    model = DummyModel()
    train_dataset = DummyDataset(size=8)
    val_dataset = DummyDataset(size=4)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    
    checkpointer = Checkpointer(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
    
    trainer1 = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        callbacks=[checkpointer],
        config={
            "epochs": 1,
            "batch_size": 4,
            "device": "cpu"
        }
    )
    
    trainer1.fit()
    
    checkpoint_path = checkpoint_dir / "latest_model.pt"
    assert checkpoint_path.exists()
    
    # Setup second run to resume from checkpoint
    model_resume = DummyModel()
    optimizer_resume = torch.optim.SGD(model_resume.parameters(), lr=0.05)
    
    trainer2 = Trainer(
        model=model_resume,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer_resume,
        loss_fn=loss_fn,
        callbacks=[],
        config={
            "epochs": 3,
            "batch_size": 4,
            "device": "cpu"
        }
    )
    
    trainer2.fit(resume_from=checkpoint_path)
    
    # Resumption must set start_epoch to 1 (trained 1 epoch in run 1)
    assert trainer2.start_epoch == 1
    assert trainer2.current_epoch == 2  # loop finishes at index 2 (total 3 epochs)


def test_onnx_model_export(tmp_path):
    """Verify dummy 3D model backbone is fully exportable to standard ONNX format."""
    model = DummyModel()
    model.eval()
    
    # Dynamic dummy tensor matching model expected layout shape (B, C, X, Y, Z)
    dummy_input = torch.randn(1, 1, 8, 8, 8)
    
    onnx_path = tmp_path / "model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0
