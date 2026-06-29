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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve, average_precision_score

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
from utils.device import resolve_backend

from training.preprocess_autism import preprocess_abide_dataset, wrap_raw_mri

logger = setup_logger("train_autism", "training/train_autism.log")


class Random3DAugmentation:
    """Applies on-the-fly random 3D augmentations to PyTorch tensors to prevent overfitting."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_std: float = 0.015,
        intensity_scale_range: tuple[float, float] = (0.9, 1.1),
        intensity_shift_range: tuple[float, float] = (-0.08, 0.08),
        translation_range: tuple[int, int] = (-4, 4)
    ):
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.scale_min, self.scale_max = intensity_scale_range
        self.shift_min, self.shift_max = intensity_shift_range
        self.trans_min, self.trans_max = translation_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        import random
        x = tensor.clone()

        # 1. Left-Right Flip (flips along dimension 1 of shape C,X,Y,Z)
        if random.random() < self.flip_prob:
            x = torch.flip(x, dims=[1])

        # 2. Random 3D Translation with zero-padding (shifts by up to N voxels)
        if random.random() < 0.5:
            shift_x = random.randint(self.trans_min, self.trans_max)
            shift_y = random.randint(self.trans_min, self.trans_max)
            shift_z = random.randint(self.trans_min, self.trans_max)
            
            x = torch.roll(x, shifts=(shift_x, shift_y, shift_z), dims=(1, 2, 3))
            
            # Zero out boundary wrap-around regions
            if shift_x > 0:
                x[:, :shift_x, :, :] = 0
            elif shift_x < 0:
                x[:, shift_x:, :, :] = 0
                
            if shift_y > 0:
                x[:, :, :shift_y, :] = 0
            elif shift_y < 0:
                x[:, :, shift_y:, :] = 0
                
            if shift_z > 0:
                x[:, :, :, :shift_z] = 0
            elif shift_z < 0:
                x[:, :, :, shift_z:] = 0

        # 3. Random Gaussian Noise
        if self.noise_std > 0 and random.random() < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # 4. Random Intensity Scaling & Shifting
        if random.random() < 0.5:
            scale = random.uniform(self.scale_min, self.scale_max)
            shift = random.uniform(self.shift_min, self.shift_max)
            x = x * scale + shift

        return x


def validate_cache_data(
    train_dataset: ABIDEDataset,
    val_dataset: ABIDEDataset,
    preprocessed_dir: Path,
    config_yaml: Path,
    full_validation: bool = False,
) -> Dict[str, Any]:
    """Validates the cache files for the active train and val split subjects."""
    from training.preprocess_autism import compute_pipeline_hash
    
    report = {
        "valid": True,
        "pipeline_hash_match": True,
        "expected_hash": "",
        "actual_hash": "",
        "subjects_checked": 0,
        "missing_subjects": [],
        "invalid_shapes": [],
        "invalid_dtypes": [],
        "corrupt_subjects": []
    }
    
    # 1. Compute expected hash
    expected_steps = []
    if config_yaml.exists():
        try:
            with open(config_yaml, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            expected_steps = cfg.get("profiles", {}).get("autism", [])
        except Exception as e:
            logger.error(f"Failed to read preprocessing config for validation: {e}")
            
    expected_hash = compute_pipeline_hash(expected_steps)
    report["expected_hash"] = expected_hash
    
    # Extract expected target shape from expected steps
    target_shape = [128, 128, 128]  # Default fallback
    for step in reversed(expected_steps):
        if step.get("transform") in ("resize", "pad", "center_crop"):
            params = step.get("params", {})
            if "target_shape" in params:
                target_shape = params["target_shape"]
                break
    expected_shape = (1,) + tuple(target_shape)
    
    # 2. Check metadata.json in cache folder
    meta_path = preprocessed_dir / "metadata.json"
    actual_hash = None
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta_data = json.load(f)
            actual_hash = meta_data.get("pipeline_hash")
            report["actual_hash"] = actual_hash or "missing_hash"
        except Exception as e:
            logger.warning(f"Failed to read cache metadata.json: {e}")
            report["actual_hash"] = "corrupt_metadata"
    else:
        report["actual_hash"] = "missing_metadata"
        
    if actual_hash != expected_hash:
        report["pipeline_hash_match"] = False
        report["valid"] = False
        logger.warning(
            f"Pipeline hash mismatch! Expected: {expected_hash}, Cached: {actual_hash}. "
            "Preprocessed cache might be stale."
        )
        
    # Collect all subject items to check
    subjects_to_check = []
    for item in train_dataset.items:
        subjects_to_check.append(item["subject_id"])
    for item in val_dataset.items:
        subjects_to_check.append(item["subject_id"])
        
    # De-duplicate
    subjects_to_check = list(set(subjects_to_check))
    report["subjects_checked"] = len(subjects_to_check)
    
    # Build a set of all files in the directory to do O(1) local checks instead of sequential exists() FUSE roundtrips
    import os
    cached_files = set()
    if preprocessed_dir.exists():
        try:
            for entry in os.scandir(preprocessed_dir):
                if entry.is_file():
                    cached_files.add(entry.name)
        except Exception as e:
            logger.warning(f"Failed to scan cache directory: {e}")
            
    for sub_id in subjects_to_check:
        pt_filename = f"{sub_id}.pt"
        pt_path = preprocessed_dir / pt_filename
        if pt_filename not in cached_files:
            report["missing_subjects"].append(sub_id)
            report["valid"] = False
            continue
            
        if not full_validation:
            continue
            
        try:
            cache = torch.load(pt_path, map_location="cpu", weights_only=False)
            tensor = cache.get("image")
            
            if tensor is None:
                report["corrupt_subjects"].append(sub_id)
                report["valid"] = False
                continue
                
            actual_shape = tuple(tensor.shape)
            if actual_shape != expected_shape:
                report["invalid_shapes"].append({
                    "subject_id": sub_id,
                    "expected": list(expected_shape),
                    "actual": list(actual_shape)
                })
                report["valid"] = False
                
            if hasattr(tensor, "dtype"):
                dtype_str = str(tensor.dtype)
                if "float32" not in dtype_str:
                    report["invalid_dtypes"].append({
                        "subject_id": sub_id,
                        "expected": "float32",
                        "actual": dtype_str
                    })
                    report["valid"] = False
            else:
                report["invalid_dtypes"].append({
                    "subject_id": sub_id,
                    "expected": "float32",
                    "actual": "unknown"
                })
                report["valid"] = False
                
        except Exception as e:
            logger.error(f"Failed loading cache for {sub_id}: {e}")
            report["corrupt_subjects"].append(sub_id)
            report["valid"] = False
            
    return report


def run_training_experiment(
    data_root: str | Path,
    index_file: str | Path,
    split_file: str | Path,
    preprocessed_dir: str | Path,
    config_yaml: str | Path,
    experiment_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 4,
    device: str = "auto",
    lr: float = 1e-3,
    resume_from: str | Path | None = "auto",
    skip_preprocess: bool = False,
    copy_outputs_to: str | Path | None = None,
    strict_cache_validation: bool = False,
    limit: int | None = None,
    full_cache_validation: bool = False,
    local_cache_dir: str | Path | None = None,
    augment: bool = False,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    scheduler_patience: int = 6,
    scheduler_factor: float = 0.5,
    seed: int = 42,
    experiment_name: str = "unnamed_experiment",
    decision_threshold: float = 0.5,
    early_stopping_patience: int = 8,
) -> None:
    """Orchestrates full train/validation, checkpointers, and classification plots reports."""
    # Set reproducibility seeds
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    index_path = Path(index_file)
    split_path = Path(split_file)
    config_path = Path(config_yaml)
    
    backend = resolve_backend(device)
    resolved_device = backend.device
    logger.info(f"Resolved device backend: {backend.name} (device: {resolved_device}, version: {backend.version})")
    
    # Load cache version dynamically
    logger.info("Loading preprocessing configuration...")
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cache_version = str(cfg.get("cache_version", "v1"))
    else:
        cache_version = "v1"
    logger.info(f"Using cache version directory: {cache_version}")
        
    preprocessed_path = Path(preprocessed_dir) / cache_version
    exp_dir = Path(experiment_dir)
    
    logger.info(f"Creating experiment output directory (Google Drive mount path): {exp_dir}...")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment directory successfully verified.")
    
    # 1. Offline Preprocessing run
    if not skip_preprocess:
        preprocess_abide_dataset(index_path, Path(preprocessed_dir), config_path, raw_dir=data_root)
    else:
        logger.info("Skipping offline dataset preprocessing as requested.")
        
    # 1.5 Handle copying cache to local SSD if requested
    if local_cache_dir is not None:
        local_cache_path = Path(local_cache_dir) / cache_version
        logger.info(f"Local cache directory configured: {local_cache_path}")
        
        # Check if cache is already copied (verify if metadata.json exists locally AND contains pt files)
        local_meta = local_cache_path / "metadata.json"
        
        import os
        local_pt_count = 0
        if local_cache_path.exists():
            try:
                local_pt_count = sum(1 for entry in os.scandir(local_cache_path) if entry.is_file() and entry.name.endswith(".pt"))
            except Exception:
                pass
                
        if not local_meta.exists() or local_pt_count < 100:
            logger.info(f"Copying preprocessed cache from {preprocessed_path} to local SSD {local_cache_path}...")
            import shutil
            import os
            from concurrent.futures import ThreadPoolExecutor
            try:
                local_cache_path.mkdir(parents=True, exist_ok=True)
                remote_meta = preprocessed_path / "metadata.json"
                if remote_meta.exists():
                    shutil.copy(remote_meta, local_meta)
                
                # Efficient directory scanning
                pt_files = []
                for entry in os.scandir(preprocessed_path):
                    if entry.is_file() and entry.name.endswith(".pt"):
                        pt_files.append(Path(entry.path))
                        
                logger.info(f"Found {len(pt_files)} cache files to copy...")
                
                def copy_single_file(src_path: Path):
                    shutil.copy(src_path, local_cache_path / src_path.name)
                
                # Copy in parallel using 16 threads to saturate FUSE network mount I/O
                copied_count = 0
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = [executor.submit(copy_single_file, f) for f in pt_files]
                    for future in futures:
                        future.result()  # blocks until completed, raises exception if failed
                        copied_count += 1
                        if copied_count % 100 == 0 or copied_count == len(pt_files):
                            logger.info(f"Copied {copied_count}/{len(pt_files)} cache files to local SSD.")
                        
                logger.info("Successfully completed copying cache to local SSD!")
            except Exception as e:
                logger.error(f"Failed to copy cache to local SSD: {e}. Falling back to remote cache.")
                local_cache_path = preprocessed_path
        else:
            logger.info("Cache files are already present on local SSD. Skipping copy step.")
            
        preprocessed_path = local_cache_path
    
    # 2. Build Datasets
    label_map = {"CONTROL": 0, "ASD": 1}
    
    # Define augmentations for training if requested
    train_transform = None
    if augment:
        train_transform = Random3DAugmentation(
            flip_prob=0.5,
            noise_std=0.015,
            intensity_scale_range=(0.9, 1.1),
            intensity_shift_range=(-0.08, 0.08),
            translation_range=(-4, 4)
        )
        logger.info("3D data augmentation enabled for training dataset.")

    train_dataset = ABIDEDataset(
        index_file=index_path,
        split_file=split_path,
        split_name="train",
        preprocessed_dir=preprocessed_path,
        raw_dir=data_root,
        label_map=label_map,
        transform=train_transform
    )
    val_dataset = ABIDEDataset(
        index_file=index_path,
        split_file=split_path,
        split_name="val",
        preprocessed_dir=preprocessed_path,
        raw_dir=data_root,
        label_map=label_map
    )
    
    if limit is not None:
        logger.info(f"Limiting train dataset to first {limit} subjects and val dataset to first {limit} subjects.")
        train_dataset.items = train_dataset.items[:limit]
        val_dataset.items = val_dataset.items[:limit]

    logger.info(f"Loaded train samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")
    
    # 2b. Perform pre-training cache validation
    validation_report = validate_cache_data(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        preprocessed_dir=preprocessed_path,
        config_yaml=config_path,
        full_validation=full_cache_validation,
    )
    
    # Save cache validation report to experiment directory
    report_path = exp_dir / "cache_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=4)
    logger.info(f"Saved cache validation report to: {report_path}")
    
    # Copy preprocessor config and cache metadata.json (provenance backups) to experiment outputs directory
    if config_path.exists():
        import shutil
        shutil.copy(config_path, exp_dir / "preprocessing.yaml")
        logger.info(f"Copied preprocessing config to: {exp_dir / 'preprocessing.yaml'}")
    
    meta_path = preprocessed_path / "metadata.json"
    if meta_path.exists():
        import shutil
        shutil.copy(meta_path, exp_dir / "cache_metadata.json")
        logger.info(f"Copied cache metadata fingerprint to: {exp_dir / 'cache_metadata.json'}")
        
    if not validation_report["valid"]:
        msg = f"Cache validation failed! Details: {validation_report}"
        if strict_cache_validation:
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.warning(msg)
    
    # 3. Model construction
    model = DenseNet3D(in_channels=1, out_channels=2)
    
    # 3b. Optimizer setup
    opt_name = optimizer_name.lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose 'adam' or 'adamw'.")
    logger.info(f"Initialized {optimizer.__class__.__name__} optimizer (lr={lr}, weight_decay={weight_decay})")
    
    loss_fn = nn.CrossEntropyLoss()
    
    # ReduceLROnPlateau Scheduler
    scheduler = get_scheduler("ReduceLROnPlateau", optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor)
    
    # Save the initial experiment configuration
    meta_path = exp_dir / "experiment_meta.json"
    meta_config = {
        "experiment_name": experiment_name,
        "architecture": "DenseNet3D",
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "early_stopping_patience": early_stopping_patience,
        "decision_threshold": decision_threshold,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "augmentation": augment,
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_config, f, indent=4)
    logger.info(f"Saved initial experiment metadata configuration to: {meta_path}")

    # Config parameters dict
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(resolved_device),
        "backend_name": str(backend.name),
        "backend_version": str(backend.version),
        "torch_version": str(torch.__version__),
        "amp": True,
        "grad_clip_max_norm": 1.0,
        "learning_rate": lr,
        "augment": augment,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "early_stopping_patience": early_stopping_patience,
        "decision_threshold": decision_threshold,
        "seed": seed,
        "experiment_name": experiment_name,
        "monitor": "val_loss",
        "mode": "min"
    }
    
    # 4. Callbacks setup
    checkpointer = Checkpointer(dirpath=exp_dir, monitor="val_loss", mode="min")
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_patience, min_delta=1e-4, mode="min")
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
    
    # Resolve auto-resume checkpoint path
    resume_path = None
    if resume_from == "auto" or resume_from == "latest":
        auto_path = exp_dir / "latest_model.pt"
        if auto_path.exists():
            resume_path = auto_path
            logger.info(f"Auto-resume: Found latest checkpoint at {resume_path}. Resuming training...")
        else:
            logger.info("Auto-resume: No previous checkpoint found in experiment directory. Starting training from scratch.")
    elif resume_from is not None and str(resume_from).lower() != "none":
        resume_path = Path(resume_from)

    # Copy best_model.pt from source directory if resuming in a new experiment directory
    if resume_path is not None and resume_path.exists():
        src_best = resume_path.parent / "best_model.pt"
        dest_best = exp_dir / "best_model.pt"
        if src_best.exists() and src_best != dest_best:
            try:
                import shutil
                shutil.copy(src_best, dest_best)
                logger.info(f"Auto-resume: Copied best checkpoint from {src_best} to {dest_best}")
            except Exception as e:
                logger.warning(f"Auto-resume: Failed to copy best checkpoint: {e}")

    # 5. Fit loop
    trainer.fit(resume_from=resume_path)
    
    # Save the exact experiment configuration
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
        
    # 6. Evaluation on validation set using the best model checkpoint
    best_ckpt_path = exp_dir / "best_model.pt"
    best_epoch = None
    best_val_loss = None
    best_val_accuracy = None
    
    if best_ckpt_path.exists():
        logger.info("Loading best model weights for final evaluation...")
        best_ckpt = torch.load(best_ckpt_path, map_location=resolved_device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        
        best_epoch = int(best_ckpt.get("epoch", 0))
        best_val_loss = float(best_ckpt.get("best_metric", 0.0))
        best_val_accuracy = float(best_ckpt.get("metrics", {}).get("val_val_accuracy", 0.0))
        logger.info(f"Loaded best checkpoint weights from epoch {best_epoch} (val_loss: {best_val_loss:.4f}, val_acc: {best_val_accuracy:.4f})")

    # Update experiment metadata with final best results
    best_val_pr_auc = None
    if best_ckpt_path.exists():
        try:
            # Recompute average precision (PR AUC) if valid checkpoint is found
            best_val_pr_auc = float(best_ckpt.get("metrics", {}).get("val_val_pr_auc", 0.0))
        except Exception:
            pass
            
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_config = json.load(f)
        except Exception:
            pass
            
    meta_config.update({
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    })
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_config, f, indent=4)
    logger.info(f"Updated experiment metadata with best epoch results at: {meta_path}")
        
    model.to(resolved_device)
    model.eval()
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if backend.capabilities.pin_memory else 0,
        pin_memory=backend.capabilities.pin_memory,
        collate_fn=collate_dataset_samples
    )
    
    y_true = []
    y_pred = []
    y_prob = []
    y_logit = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(resolved_device, non_blocking=backend.capabilities.non_blocking)
            targets = batch["label"].to(resolved_device, non_blocking=backend.capabilities.non_blocking)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            probs_asd = probs[:, 1]
            preds = (probs_asd >= decision_threshold).long()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            y_logit.extend(outputs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    y_logit = np.array(y_logit)
    
    # Handle single target class cases gracefully for evaluation plotting
    avg_precision = None
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
        
        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob[:, 1])
        avg_precision = average_precision_score(y_true, y_prob[:, 1])
        
        plt.figure()
        plt.plot(recall_vals, precision_vals, color="blue", lw=2, label=f"PR curve (AP = {avg_precision:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(exp_dir / "pr_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved PR curve plot (AP = {avg_precision:.4f}).")

        # Probability Histogram of ASD predictions (class 1) stratified by True Label
        try:
            plt.figure()
            asd_probs_control = y_prob[y_true == 0, 1]
            asd_probs_asd = y_prob[y_true == 1, 1]
            
            plt.hist(asd_probs_control, bins=10, alpha=0.5, label="Control", color="blue", edgecolor="black")
            plt.hist(asd_probs_asd, bins=10, alpha=0.5, label="Autism (ASD)", color="orange", edgecolor="black")
            plt.xlabel("Predicted Probability of ASD")
            plt.ylabel("Count")
            plt.title("Distribution of Predicted ASD Probabilities")
            plt.legend(loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(exp_dir / "probability_histogram.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved probability histogram plot.")
        except Exception as e:
            logger.error(f"Failed to generate probability histogram: {e}")
            
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
        logger.warning("Val dataset contains only one label class. Skipping ROC, PR, Histogram, and Confusion Matrix plots.")

    # Save detailed predictions to CSV file
    try:
        import csv
        pred_csv_path = exp_dir / "predictions.csv"
        inv_label_map = {0: "Control", 1: "Autism"}
        
        with open(pred_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Subject",
                "True_Label",
                "Pred_Label",
                "Probability_ASD",
                "Probability_Control",
                "Logit_ASD",
                "Logit_Control",
                "Correct"
            ])
            
            for idx, item in enumerate(val_dataset.items):
                subj = item["subject_id"]
                true_lbl = int(y_true[idx])
                pred_lbl = int(y_pred[idx])
                
                true_label_str = inv_label_map.get(true_lbl, str(true_lbl))
                pred_label_str = inv_label_map.get(pred_lbl, str(pred_lbl))
                
                prob_asd = float(y_prob[idx, 1])
                prob_control = float(y_prob[idx, 0])
                
                logit_asd = float(y_logit[idx, 1])
                logit_control = float(y_logit[idx, 0])
                
                correct_bool = bool(true_lbl == pred_lbl)
                
                writer.writerow([
                    subj,
                    true_label_str,
                    pred_label_str,
                    f"{prob_asd:.4f}",
                    f"{prob_control:.4f}",
                    f"{logit_asd:.4f}",
                    f"{logit_control:.4f}",
                    str(correct_bool)
                ])
                
        logger.info(f"Saved validation predictions to: {pred_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions.csv: {e}")

    # Append PR AUC details to meta config if computed
    if avg_precision is not None:
        best_val_pr_auc = float(avg_precision)
        
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_config = json.load(f)
        except Exception:
            pass
            
    meta_config.update({
        "best_val_pr_auc": best_val_pr_auc
    })
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_config, f, indent=4)
        
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
            opset_version=18,
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
    parser.add_argument("--device", default="auto", help="Target execution device (e.g. cpu, cuda, directml, auto)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--resume-from", default="auto", help="Resume training checkpoint path, 'auto' to auto-detect latest, or 'none' to start from scratch")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip offline dataset preprocessing validation/run step")
    parser.add_argument("--copy-outputs-to", default=None, help="Optional directory to copy final experiment outputs back to (e.g. on Google Drive)")
    parser.add_argument("--strict-cache-validation", action="store_true", help="Halt training immediately if cache validation fails")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size to first N subjects for debug/test training")
    parser.add_argument("--full-cache-validation", action="store_true", help="Perform full cache validation by loading every cached file (slow on Google Drive)")
    parser.add_argument("--local-cache-dir", default=None, help="Local SSD directory override to copy cache once to before training")
    parser.add_argument("--augment", action="store_true", help="Enable 3D data augmentation on the training set to prevent overfitting")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"], help="Optimizer choice (adam or adamw)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--scheduler-patience", type=int, default=6, help="Patience epochs before decaying learning rate")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Multiplication factor for learning rate decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-name", default="unnamed_experiment", help="Name to identify this training experiment run")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Decision threshold for class probability prediction")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Number of validation epochs before early stopping")

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
        strict_cache_validation=args.strict_cache_validation,
        limit=args.limit,
        full_cache_validation=args.full_cache_validation,
        local_cache_dir=args.local_cache_dir,
        augment=args.augment,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        seed=args.seed,
        experiment_name=args.experiment_name,
        decision_threshold=args.decision_threshold,
        early_stopping_patience=args.early_stopping_patience,
    )
