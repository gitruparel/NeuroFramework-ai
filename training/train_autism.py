"""Autism disease training executable coordinating preprocessing, model fitting, and validation plotting."""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from core.logging import setup_logger
from preprocessing.pipeline import PreprocessingPipeline
from schemas.mri import MRIData, RawMRI
from schemas.quality import QualityReport
from schemas.validation import ValidationReport, FileValidationReport, MRIValidationReport
from engine.metadata import MetadataExtractor
from schemas.processing import ExecutionContext
from schemas.dataset import collate_dataset_samples
from datasets.abide import ABIDEDataset
from models.densenet import DenseNet3D
from training.trainer import Trainer
from training.checkpoint import Checkpointer
from training.callbacks import EarlyStopping, HistoryTracker, WandbLogger
from training.scheduler import get_scheduler

from training.preprocess_autism import preprocess_abide_dataset, wrap_raw_mri

logger = setup_logger("train_autism", "training/train_autism.log")


def run_training_experiment(
    data_root: str | Path,
    index_file: str | Path,
    split_file: str | Path,
    preprocessed_dir: str | Path,
    config_yaml: str | Path,
    experiment_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 4,
    device: str = "cpu",
    lr: float = 1e-3,
    resume_from: str | Path | None = None,
    skip_preprocess: bool = False,
    copy_outputs_to: str | Path | None = None,
) -> None:
    """Orchestrates full train/validation, checkpointers, and classification plots reports."""
    index_path = Path(index_file)
    split_path = Path(split_file)
    config_path = Path(config_yaml)
    
    # Load cache version dynamically
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cache_version = str(cfg.get("cache_version", "v1"))
    else:
        cache_version = "v1"
        
    preprocessed_path = Path(preprocessed_dir) / cache_version
    exp_dir = Path(experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Offline Preprocessing run
    if not skip_preprocess:
        preprocess_abide_dataset(index_path, Path(preprocessed_dir), config_path, raw_dir=data_root)
    else:
        logger.info("Skipping offline dataset preprocessing as requested.")
    
    # 2. Build Datasets
    label_map = {"CONTROL": 0, "ASD": 1}
    train_dataset = ABIDEDataset(
        index_file=index_path,
        split_file=split_path,
        split_name="train",
        preprocessed_dir=preprocessed_path,
        raw_dir=data_root,
        label_map=label_map
    )
    val_dataset = ABIDEDataset(
        index_file=index_path,
        split_file=split_path,
        split_name="val",
        preprocessed_dir=preprocessed_path,
        raw_dir=data_root,
        label_map=label_map
    )
    
    logger.info(f"Loaded train samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")
    
    # 3. Model construction
    model = DenseNet3D(in_channels=1, out_channels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # ReduceLROnPlateau Scheduler
    scheduler = get_scheduler("ReduceLROnPlateau", optimizer, mode="min", patience=3, factor=0.5)
    
    # Config parameters dict
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "amp": True,
        "grad_clip_max_norm": 1.0,
        "learning_rate": lr,
        "monitor": "val_loss",
        "mode": "min"
    }
    
    # 4. Callbacks setup
    checkpointer = Checkpointer(dirpath=exp_dir, monitor="val_loss", mode="min")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min")
    history_tracker = HistoryTracker(output_dir=exp_dir)
    wandb_logger = WandbLogger(project="abide_autism", config=config, mode="disabled")
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        callbacks=[checkpointer, early_stopping, history_tracker, wandb_logger],
        config=config
    )
    
    # 5. Fit loop
    trainer.fit(resume_from=resume_from)
    
    # Save the exact experiment configuration
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
        
    # 6. Evaluation on validation set using the best model checkpoint
    best_ckpt_path = exp_dir / "best_model.pt"
    if best_ckpt_path.exists():
        logger.info("Loading best model weights for final evaluation...")
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        
    model.to(device)
    model.eval()
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if device == "cuda" else 0,
        pin_memory=device == "cuda",
        collate_fn=collate_dataset_samples
    )
    
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Handle single target class cases gracefully for evaluation plotting
    if len(np.unique(y_true)) > 1:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(exp_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Control", "Autism"])
        plt.yticks(tick_marks, ["Control", "Autism"])
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        # Add labels
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
        plt.savefig(exp_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        logger.warning("Val dataset contains only one label class. Skipping ROC and Confusion Matrix plots.")
        
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=["Control", "Autism"] if len(np.unique(y_true)) > 1 else None, output_dict=True, zero_division=0)
    with open(exp_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    # 7. ONNX Model Auto-Export
    try:
        onnx_path = exp_dir / "best_model.onnx"
        sample_shape = train_dataset[0].image.shape # e.g. (1, 128, 128, 128)
        dummy_shape = (1,) + tuple(sample_shape)
        dummy_input = torch.randn(dummy_shape).to(device)
        logger.info(f"Auto-exporting best checkpoint model to ONNX: {onnx_path}")
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )
        logger.info("ONNX auto-export completed successfully.")
    except Exception as e:
        logger.error(f"ONNX model export failed: {e}")
        
    # 8. Copy outputs back to destination (e.g. Google Drive) if requested
    if copy_outputs_to is not None:
        dest_dir = Path(copy_outputs_to)
        logger.info(f"Copying final experiment outputs to destination: {dest_dir}")
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            for item in exp_dir.iterdir():
                if item.is_file():
                    shutil.copy(item, dest_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
            logger.info("Successfully copied final experiment outputs.")
        except Exception as e:
            logger.error(f"Failed to copy final experiment outputs: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train 3D DenseNet model on Autism dataset.")
    parser.add_argument("--data-root", default="/content/drive/MyDrive/NeuroFramework", help="Dataset root directory")
    parser.add_argument("--index-file", default="data/abide_index.json", help="Index file JSON path")
    parser.add_argument("--split-file", default="data/abide_splits.json", help="Split file JSON path")
    parser.add_argument("--preprocessed-dir", default="/content/drive/MyDrive/NeuroFramework/cache/abide", help="Preprocessed cache directory")
    parser.add_argument("--config-yaml", default="configs/preprocessing.yaml", help="Path to preprocessing configuration YAML")
    parser.add_argument("--experiment-dir", default="/content/drive/MyDrive/NeuroFramework/experiments/autism_densenet", help="Experiment outputs directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--device", default="cuda", help="Target execution device (e.g. cpu, cuda)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--resume-from", default=None, help="Resume training checkpoint model path")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip offline dataset preprocessing validation/run step")
    parser.add_argument("--copy-outputs-to", default=None, help="Optional directory to copy final experiment outputs back to (e.g. on Google Drive)")

    args = parser.parse_args()

    run_training_experiment(
        data_root=args.data_root,
        index_file=args.index_file,
        split_file=args.split_file,
        preprocessed_dir=args.preprocessed_dir,
        config_yaml=args.config_yaml,
        experiment_dir=args.experiment_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        resume_from=args.resume_from,
        skip_preprocess=args.skip_preprocess,
        copy_outputs_to=args.copy_outputs_to,
    )
