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

logger = setup_logger("train_autism", "training/train_autism.log")


def wrap_raw_mri(raw_mri: RawMRI) -> MRIData:
    """Helper wrapping raw NIfTI scans into the MRIData schema without slow QA checks."""
    metadata = MetadataExtractor().extract(raw_mri)
    return MRIData(
        raw=raw_mri,
        metadata=metadata,
        quality=QualityReport(
            noise_score=0.0, blur_score=0.0, motion_score=0.0, contrast_score=0.0,
            dynamic_range=0.0, resolution=1.0, slice_count=raw_mri.tensor.shape[-1], overall_score=1.0
        ),
        validation=ValidationReport(
            file_validation=FileValidationReport(exists=True, readable=True, header_valid=True, corrupt=False),
            mri_validation=MRIValidationReport(
                voxel_spacing_valid=True, dimensions_valid=True, intensity_valid=True,
                orientation_valid=True, metadata_complete=True, empty_slices_detected=False
            ),
            is_valid=True
        ),
        history=[],
        preview=[],
        statistics={"min": 0.0, "max": 1.0, "mean": 0.0, "std": 1.0, "shape": list(raw_mri.tensor.shape), "dtype": str(raw_mri.tensor.dtype)}
    )


def preprocess_abide_dataset(index_file: Path, preprocessed_dir: Path, config_yaml: Path) -> None:
    """Preprocesses all raw ABIDE scans once and caches the outputs as .pt tensors to disk."""
    logger.info("Starting one-time offline dataset preprocessing...")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pipeline
    pipeline = PreprocessingPipeline.from_yaml(config_yaml, "autism")
    
    # Context
    ctx = ExecutionContext(
        logger=logger,
        cache=None,
        config={},
        seed=42,
        device="cpu"
    )
    
    with open(index_file, encoding="utf-8") as f:
        items = json.load(f)
        
    from engine.readers.nifti import NiftiReader
    reader = NiftiReader()
    
    for idx, item in enumerate(items):
        subject_id = item["subject_id"]
        path_str = item["path"]
        
        cache_path = preprocessed_dir / f"{subject_id}.pt"
        if cache_path.exists():
            continue
            
        logger.info(f"[{idx+1}/{len(items)}] Preprocessing subject: {subject_id}")
        try:
            # Fast raw read
            raw_mri = reader.read(Path(path_str))
            mri_data = wrap_raw_mri(raw_mri)
            
            # Execute pipeline
            processed = pipeline.process(mri_data, ctx)
            
            # Save preprocessed outputs
            torch.save({
                "image": processed.image,
                "affine": processed.affine,
                "metadata": processed.metadata.model_dump()
            }, cache_path)
            
        except Exception as e:
            logger.error(f"Failed preprocessing for subject {subject_id}: {e}")
            
    logger.info("Offline dataset preprocessing completed successfully.")


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
    resume_from: str | Path | None = None
) -> None:
    """Orchestrates full train/validation, checkpointers, and classification plots reports."""
    index_path = Path(index_file)
    split_path = Path(split_file)
    preprocessed_path = Path(preprocessed_dir)
    config_path = Path(config_yaml)
    exp_dir = Path(experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Offline Preprocessing run
    preprocess_abide_dataset(index_path, preprocessed_path, config_path)
    
    # 2. Build Datasets
    label_map = {"CONTROL": 0, "ASD": 1}
    train_dataset = ABIDEDataset(
        index_file=index_path,
        split_file=split_path,
        split_name="train",
        preprocessed_dir=preprocessed_path,
        label_map=label_map
    )
    val_dataset = ABIDEDataset(
        index_file=index_path,
        split_file=split_path,
        split_name="val",
        preprocessed_dir=preprocessed_path,
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
        "amp": False, # disable amp on CPU to prevent test suite warnings
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dataset_samples)
    
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)
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
        dummy_input = torch.randn(1, 1, 32, 32, 32).to(device)
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


if __name__ == "__main__":
    # Setup baseline run params for Colab/Drive training
    run_training_experiment(
        data_root="/content/drive/MyDrive/NeuroFramework",
        index_file="data/abide_index.json",
        split_file="data/abide_splits.json",
        preprocessed_dir="/content/drive/MyDrive/NeuroFramework/cache/abide",
        config_yaml="configs/preprocessing.yaml",
        experiment_dir="/content/drive/MyDrive/NeuroFramework/experiments/autism_densenet",
        epochs=5,
        batch_size=2,
        device="cuda",
        lr=1e-3,
    )
