"""Autism disease training executable coordinating preprocessing, model fitting, and validation plotting."""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import yaml
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve, average_precision_score, f1_score, balanced_accuracy_score

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
    dropout_prob: float = 0.0,
    label_smoothing: float = 0.0,
    use_class_weights: bool = False,
    is_hyperopt: bool = False,
    augmentation_profile: str = "moderate",
    architecture: str = "densenet121",
    loss_function: str = "ce",
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
    tta: bool = False,
    tta_runs: int = 5,
    tta_method: str = "mean",
    pretrained: bool = False,
    pretrained_source: str = "none",
    pretrained_checkpoint: Optional[str] = None,
    freeze_backbone: bool = False,
    freeze_epochs: int = 0,
    backbone_lr: Optional[float] = None,
    classifier_lr: Optional[float] = None,
) -> Dict[str, Any]:
    """Orchestrates full train/validation, checkpointers, and classification plots reports."""
    # Set reproducibility seeds
    from training.experiment import set_seed
    set_seed(seed)
    
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
        from training.augmentations import get_mri_augmentations
        train_transform = get_mri_augmentations(profile=augmentation_profile, seed=seed)
        logger.info(f"MONAI 3D data augmentation enabled using profile '{augmentation_profile}'.")

    train_dataset = ABIDEDataset(
        index_file=index_path,
        split_file=split_path,
        split_name="train",
        preprocessed_dir=preprocessed_path,
        raw_dir=data_root,
        label_map=label_map,
        transform=train_transform
    )
    
    # Generate augmentation preview if requested and not in hyperopt trial mode
    if augment and not is_hyperopt:
        try:
            from training.augmentations import generate_augmentation_preview
            preview_path = exp_dir / "augmentation_preview.png"
            
            # Temporarily deactivate dataset transform to read the raw preprocessed volume
            train_dataset.transform = None
            original_sample = train_dataset[0]
            train_dataset.transform = train_transform
            
            generate_augmentation_preview(original_sample.image, train_transform, preview_path)
            logger.info(f"Generated data augmentation visual preview at: {preview_path}")
        except Exception as e:
            logger.error(f"Failed to generate data augmentation visual preview: {e}")
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
    
    # 3. Model construction using ModelFactory
    from models.factory import ModelFactory
    model = ModelFactory.create_model(
        model_name=architecture,
        in_channels=1,
        out_channels=2,
        dropout_prob=dropout_prob
    )
    
    # Load pretrained weights if requested
    if pretrained or pretrained_source != "none":
        from models.pretrained import load_pretrained_weights
        model = load_pretrained_weights(
            model=model,
            source=pretrained_source,
            checkpoint_path=pretrained_checkpoint,
            architecture=architecture
        )
        
    if freeze_backbone and freeze_epochs > 0:
        from models.pretrained import freeze_backbone as freeze_bb
        freeze_bb(model)
    
    # Estimate parameter counts and model size in MB
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    # Save model summary file (model_summary.txt)
    if not is_hyperopt:
        try:
            from training.benchmark import generate_model_summary
            summary_path = exp_dir / "model_summary.txt"
            sample_shape = train_dataset[0].image.shape  # (C, D, H, W)
            input_shape = (1,) + tuple(sample_shape)     # (1, C, D, H, W)
            generate_model_summary(model, input_shape, summary_path)
        except Exception as e:
            logger.error(f"Failed to generate model summary: {e}")

    # Move model to resolved device BEFORE setting up optimizer parameters
    model.to(resolved_device)
    
    # Enable DataParallel if multi-GPU is available on CUDA
    if resolved_device.type == "cuda" and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with nn.DataParallel!")
        model = torch.nn.DataParallel(model)
        
    # 3b. Optimizer setup
    opt_name = optimizer_name.lower()
    
    # Setup parameters with differential learning rates if requested
    classifier_names = ["class_out", "classifier", "fc"]
    backbone_params = []
    classifier_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if any(cn in n for cn in classifier_names):
                classifier_params.append(p)
            else:
                backbone_params.append(p)
                
    param_groups = []
    if backbone_lr is not None and classifier_lr is not None:
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if classifier_params:
            param_groups.append({"params": classifier_params, "lr": classifier_lr})
        logger.info(f"Using differential learning rates: backbone_lr={backbone_lr}, classifier_lr={classifier_lr}")
    else:
        # Use single learning rate
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
        logger.info(f"Using single learning rate: lr={lr}")
        
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose 'adam' or 'adamw'.")
    logger.info(f"Initialized {optimizer.__class__.__name__} optimizer (weight_decay={weight_decay})")
    
    # Measure class distribution and ratio
    train_labels = [int(item["label"] == "ASD") for item in train_dataset.items]
    class_counts = np.bincount(train_labels)
    num_control = int(class_counts[0]) if len(class_counts) > 0 else 0
    num_asd = int(class_counts[1]) if len(class_counts) > 1 else 0
    ratio = max(num_control, num_asd) / max(min(num_control, num_asd), 1)
    logger.info(f"Training split distribution - Control: {num_control}, Autism: {num_asd}, Imbalance Ratio: {ratio:.2f}")
    
    # Setup class weights if requested
    loss_weight_tensor = None
    if use_class_weights or loss_function in ["weighted_ce", "weighted-ce"]:
        total_samples = len(train_labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        loss_weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(resolved_device)
        logger.info(f"Applying class-weighted loss with weights: {class_weights}")
        
    # Override with focal_alpha balance if provided for Focal Loss
    if loss_function in ["focal", "focal_ls"] and focal_alpha is not None:
        loss_weight_tensor = torch.tensor([focal_alpha, 1.0 - focal_alpha], dtype=torch.float).to(resolved_device)
        logger.info(f"Applying Focal Loss alpha balance: [{focal_alpha}, {1.0 - focal_alpha}]")
        
    from training.losses import LossFactory
    loss_fn = LossFactory.create_loss(
        loss_name=loss_function,
        alpha=loss_weight_tensor,
        gamma=focal_gamma,
        label_smoothing=label_smoothing
    )
    logger.info(f"Using loss function: {loss_function}")
    
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
        "dropout_prob": dropout_prob,
        "label_smoothing": label_smoothing,
        "use_class_weights": use_class_weights or (loss_function == "weighted_ce"),
        "loss_function": loss_function,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "initialization_type": pretrained_source,
        "pretrained_source": pretrained_source,
        "frozen_epochs": freeze_epochs,
        "backbone_learning_rate": backbone_lr,
        "classifier_learning_rate": classifier_lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "augmentation": augment,
        "architecture": architecture,
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "model_size_mb": model_size_mb,
        "backbone_name": architecture,
    }
    
    if augment:
        from training.augmentations import get_profile_metadata
        aug_meta = get_profile_metadata(augmentation_profile)
        meta_config.update({
            "augmentation_profile": augmentation_profile,
            "enabled_transforms": aug_meta["enabled_transforms"],
            "transform_probabilities": aug_meta["transform_probabilities"],
            "random_seed": seed
        })
    else:
        meta_config.update({
            "augmentation_profile": "none",
            "enabled_transforms": [],
            "transform_probabilities": {},
            "random_seed": seed
        })
        
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
        "dropout_prob": dropout_prob,
        "label_smoothing": label_smoothing,
        "use_class_weights": use_class_weights,
        "seed": seed,
        "experiment_name": experiment_name,
        "monitor": "val_loss",
        "mode": "min",
        "freeze_epochs": freeze_epochs,
        "backbone_lr": backbone_lr,
        "classifier_lr": classifier_lr,
        "pretrained_source": pretrained_source,
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

    # 5. Fit loop with training time monitoring
    import time
    start_time = time.time()
    trainer.fit(resume_from=resume_path)
    training_time = time.time() - start_time
    
    actual_epochs = len(history_tracker.history.get("epoch", []))
    epoch_time = training_time / max(1, actual_epochs)
    logger.info(f"Training completed in {training_time:.2f}s (avg {epoch_time:.2f}s/epoch)")
    
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
        from utils.device import load_state_dict_flexible
        load_state_dict_flexible(model, best_ckpt["model_state_dict"])
        
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
        "best_val_pr_auc": best_val_pr_auc,
        "training_time": training_time,
        "epoch_time": epoch_time,
    })
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_config, f, indent=4)
    logger.info(f"Updated experiment metadata with best epoch results at: {meta_path}")
        
    model.to(resolved_device)
    model.eval()
    
    # Wrap in DataParallel for validation speedup if available
    eval_model = model
    if resolved_device.type == "cuda" and torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
        logger.info(f"Wrapping validation model in nn.DataParallel to use {torch.cuda.device_count()} GPUs!")
        eval_model = torch.nn.DataParallel(model)
        
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if backend.capabilities.pin_memory else 0,
        pin_memory=backend.capabilities.pin_memory,
        collate_fn=collate_dataset_samples
    )
    
    # Define a robust inference runner supporting Test-Time Augmentation
    def run_inference_loop(
        m: nn.Module,
        loader: DataLoader,
        dev: torch.device,
        cap_backend,
        tta_aug = None,
        runs: int = 1,
        method: str = "mean"
    ):
        y_t = []
        y_p = []
        y_pr = []
        y_lg = []
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(dev, non_blocking=cap_backend.capabilities.non_blocking)
                targets = batch["label"].to(dev, non_blocking=cap_backend.capabilities.non_blocking)
                
                if tta_aug is not None and runs > 1:
                    batch_probs = []
                    batch_logits = []
                    for r in range(runs):
                        # Augment each spatial volume individually and restack
                        augmented_inputs = []
                        for i in range(inputs.size(0)):
                            augmented_inputs.append(tta_aug.get_augmentations(inputs[i], r))
                        augmented_batch = torch.stack(augmented_inputs, dim=0)
                        
                        outputs = m(augmented_batch)
                        probs = torch.softmax(outputs, dim=1)
                        batch_probs.append(probs)
                        batch_logits.append(outputs)
                        
                    from training.inference import PredictionAggregator
                    aggregated_probs = PredictionAggregator.aggregate(batch_probs, method=method)
                    preds = (aggregated_probs[:, 1] >= decision_threshold).long()
                    
                    y_t.extend(targets.cpu().numpy())
                    y_p.extend(preds.cpu().numpy())
                    y_pr.extend(aggregated_probs.cpu().numpy())
                    avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
                    y_lg.extend(avg_logits.cpu().numpy())
                else:
                    outputs = m(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = (probs[:, 1] >= decision_threshold).long()
                    
                    y_t.extend(targets.cpu().numpy())
                    y_p.extend(preds.cpu().numpy())
                    y_pr.extend(probs.cpu().numpy())
                    y_lg.extend(outputs.cpu().numpy())
                    
        return np.array(y_t), np.array(y_p), np.array(y_pr), np.array(y_lg)

    import time
    from training.inference import TestTimeAugmentor, aggregate_tta_comparison
    
    # 1. Run Baseline validation inference
    start_baseline = time.time()
    y_true, y_pred, y_prob, y_logit = run_inference_loop(
        m=eval_model,
        loader=val_loader,
        dev=resolved_device,
        cap_backend=backend,
        tta_aug=None,
        runs=1,
        method="mean"
    )
    baseline_latency = time.time() - start_baseline
    
    # Calculate baseline Sensitivity/Specificity for delta tracking
    baseline_sensitivity = 0.0
    baseline_specificity = 0.0
    if len(np.unique(y_true)) > 1:
        cm_base = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm_base.ravel()
        baseline_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        baseline_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
    baseline_accuracy = float(np.mean(y_true == y_pred))
    
    # Calculate baseline ROC-AUC and PR-AUC
    from sklearn.metrics import average_precision_score, roc_auc_score
    baseline_roc_auc = 0.5
    baseline_pr_auc = 0.5
    if len(np.unique(y_true)) > 1:
        baseline_roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        baseline_pr_auc = float(average_precision_score(y_true, y_prob[:, 1]))
        
    # Calculate baseline Balanced Accuracy & Macro F1
    baseline_bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    report_base = classification_report(y_true, y_pred, target_names=["Control", "Autism"] if len(np.unique(y_true)) > 1 else None, output_dict=True, zero_division=0)
    baseline_f1 = float(report_base.get('macro avg', {}).get('f1-score', 0.0))
    
    baseline_metrics_dict = {
        "accuracy": baseline_accuracy,
        "roc_auc": baseline_roc_auc,
        "pr_auc": baseline_pr_auc,
        "macro_f1": baseline_f1,
        "balanced_accuracy": baseline_bal_acc,
        "sensitivity": baseline_sensitivity,
        "specificity": baseline_specificity
    }
    
    # 2. Run TTA validation inference if requested
    tta_metrics_dict = {}
    tta_latency = 0.0
    
    if tta and tta_runs > 1:
        logger.info(f"Running Test-Time Augmentation (TTA) with {tta_runs} runs using aggregation: {tta_method}")
        start_tta = time.time()
        y_true_tta, y_pred_tta, y_prob_tta, y_logit_tta = run_inference_loop(
            m=eval_model,
            loader=val_loader,
            dev=resolved_device,
            cap_backend=backend,
            tta_aug=TestTimeAugmentor(seed=seed),
            runs=tta_runs,
            method=tta_method
        )
        tta_latency = time.time() - start_tta
        
        # Compute TTA metrics
        tta_sensitivity = 0.0
        tta_specificity = 0.0
        if len(np.unique(y_true_tta)) > 1:
            cm_tta = confusion_matrix(y_true_tta, y_pred_tta)
            tn, fp, fn, tp = cm_tta.ravel()
            tta_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            tta_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
        tta_accuracy = float(np.mean(y_true_tta == y_pred_tta))
        tta_roc_auc = 0.5
        tta_pr_auc = 0.5
        if len(np.unique(y_true_tta)) > 1:
            tta_roc_auc = float(roc_auc_score(y_true_tta, y_prob_tta[:, 1]))
            tta_pr_auc = float(average_precision_score(y_true_tta, y_prob_tta[:, 1]))
            
        tta_bal_acc = float(balanced_accuracy_score(y_true_tta, y_pred_tta))
        report_tta = classification_report(y_true_tta, y_pred_tta, target_names=["Control", "Autism"] if len(np.unique(y_true_tta)) > 1 else None, output_dict=True, zero_division=0)
        tta_f1 = float(report_tta.get('macro avg', {}).get('f1-score', 0.0))
        
        tta_metrics_dict = {
            "accuracy": tta_accuracy,
            "roc_auc": tta_roc_auc,
            "pr_auc": tta_pr_auc,
            "macro_f1": tta_f1,
            "balanced_accuracy": tta_bal_acc,
            "sensitivity": tta_sensitivity,
            "specificity": tta_specificity
        }
        
        # Save TTA comparison reports
        aggregate_tta_comparison(baseline_metrics_dict, tta_metrics_dict, baseline_latency, tta_latency, exp_dir)
        
        # Override baseline evaluation arrays with the TTA predictions
        y_true = y_true_tta
        y_pred = y_pred_tta
        y_prob = y_prob_tta
        y_logit = y_logit_tta
        
        metric_improvements = {
            "accuracy_delta": tta_accuracy - baseline_accuracy,
            "roc_auc_delta": tta_roc_auc - baseline_roc_auc,
            "pr_auc_delta": tta_pr_auc - baseline_pr_auc,
            "macro_f1_delta": tta_f1 - baseline_f1,
            "balanced_accuracy_delta": tta_bal_acc - baseline_bal_acc,
            "sensitivity_delta": tta_sensitivity - baseline_sensitivity,
            "specificity_delta": tta_specificity - baseline_specificity
        }
    else:
        metric_improvements = {}
        
    # Calculate final Sensitivity and Specificity at the active decision threshold (TTA or Baseline)
    sensitivity = 0.0
    specificity = 0.0
    if len(np.unique(y_true)) > 1:
        cm_eval = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm_eval.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        logger.info(f"Final Validation Sensitivity (Recall): {sensitivity:.4f} | Specificity: {specificity:.4f}")
    
    # Handle single target class cases gracefully for evaluation plotting
    avg_precision = None
    roc_auc = None
    if len(np.unique(y_true)) > 1:
        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        if not is_hyperopt:
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

            # Save ROC points to CSV
            try:
                import csv
                roc_csv_path = exp_dir / "roc_points.csv"
                with open(roc_csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["threshold", "tpr", "fpr"])
                    for f_val, t_val, th_val in zip(fpr, tpr, roc_thresholds):
                        writer.writerow([f"{th_val:.4f}", f"{t_val:.4f}", f"{f_val:.4f}"])
                logger.info(f"Saved ROC points to: {roc_csv_path}")
            except Exception as e:
                logger.error(f"Failed to save roc_points.csv: {e}")
        
        # Precision-Recall Curve
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_prob[:, 1])
        avg_precision = average_precision_score(y_true, y_prob[:, 1])
        
        if not is_hyperopt:
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

            # Save PR points to CSV
            try:
                import csv
                pr_csv_path = exp_dir / "pr_points.csv"
                with open(pr_csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["threshold", "precision", "recall"])
                    for idx, th_val in enumerate(pr_thresholds):
                        writer.writerow([f"{th_val:.4f}", f"{precision_vals[idx]:.4f}", f"{recall_vals[idx]:.4f}"])
                logger.info(f"Saved PR points to: {pr_csv_path}")
            except Exception as e:
                logger.error(f"Failed to save pr_points.csv: {e}")

        if not is_hyperopt:
            # Probability Histogram of ASD predictions (class 1) stratified by True Label
            try:
                plt.figure()
                asd_probs_control = y_prob[y_true == 0, 1]
                asd_probs_asd = y_prob[y_true == 1, 1]
                
                bins_arr = np.linspace(0.0, 1.0, 11)
                plt.hist(asd_probs_control, bins=bins_arr, alpha=0.5, label="Control", color="blue", edgecolor="black")
                plt.hist(asd_probs_asd, bins=bins_arr, alpha=0.5, label="Autism (ASD)", color="orange", edgecolor="black")
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
        if not is_hyperopt:
            logger.warning("Val dataset contains only one label class. Skipping ROC, PR, Histogram, and Confusion Matrix plots.")

    if not is_hyperopt:
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

        # Save probabilities and logits as numpy binaries
        try:
            prob_path = exp_dir / "val_probabilities.npy"
            logit_path = exp_dir / "val_logits.npy"
            np.save(prob_path, y_prob)
            np.save(logit_path, y_logit)
            logger.info(f"Saved validation probability and logit arrays to: {prob_path}, {logit_path}")
        except Exception as e:
            logger.error(f"Failed to save validation numpy arrays: {e}")

    # Append PR AUC details to meta config if computed
    best_val_pr_auc = None
    if avg_precision is not None:
        best_val_pr_auc = float(avg_precision)
        
    # Compute optimal thresholds maximizing F1, Balanced Accuracy, and Youden's J
    best_f1_th = 0.5
    best_f1_val = 0.0
    best_bal_acc_th = 0.5
    best_bal_acc_val = 0.0
    best_youden_th = 0.5
    best_youden_val = -1.0
    
    if len(np.unique(y_true)) > 1:
        for th in np.linspace(0.01, 0.99, 99):
            th_preds = (y_prob[:, 1] >= th).astype(int)
            
            # F1 Score (Autism / class 1)
            th_f1 = f1_score(y_true, th_preds, zero_division=0)
            if th_f1 > best_f1_val:
                best_f1_val = th_f1
                best_f1_th = float(th)
                
            # Balanced Accuracy
            th_bal_acc = balanced_accuracy_score(y_true, th_preds)
            if th_bal_acc > best_bal_acc_val:
                best_bal_acc_val = th_bal_acc
                best_bal_acc_th = float(th)
                
            # Youden's J
            cm_th = confusion_matrix(y_true, th_preds)
            tn, fp, fn, tp = cm_th.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            th_youden = sensitivity + specificity - 1.0
            if th_youden > best_youden_val:
                best_youden_val = th_youden
                best_youden_th = float(th)
                
        logger.info(f"Optimal threshold (F1): {best_f1_th:.2f} (F1: {best_f1_val:.4f})")
        logger.info(f"Optimal threshold (Balanced Acc): {best_bal_acc_th:.2f} (Acc: {best_bal_acc_val:.4f})")
        logger.info(f"Optimal threshold (Youden's J): {best_youden_th:.2f} (J: {best_youden_val:.4f})")
        
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=["Control", "Autism"] if len(np.unique(y_true)) > 1 else None, output_dict=True, zero_division=0)
    if not is_hyperopt:
        with open(exp_dir / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_config = json.load(f)
            except Exception:
                pass
                
        # Compute validation balanced accuracy at current decision threshold
        val_bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        
        # Calculate macro f1 if report exists
        macro_f1 = report.get('macro avg', {}).get('f1-score', 0.0) if 'report' in locals() else None
        
        meta_config.update({
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "best_val_pr_auc": best_val_pr_auc,
            "best_val_roc_auc": roc_auc if (len(np.unique(y_true)) > 1 and 'roc_auc' in locals()) else None,
            "best_val_balanced_accuracy": val_bal_acc,
            "best_val_macro_f1": macro_f1,
            "best_val_sensitivity": sensitivity,
            "best_val_specificity": specificity,
            "tta_enabled": tta,
            "tta_runs": tta_runs,
            "tta_method": tta_method,
            "inference_time_increase": float(tta_latency - baseline_latency) if tta else 0.0,
            "metric_improvements": metric_improvements,
            "optimal_threshold_f1": best_f1_th,
            "optimal_threshold_f1_val": best_f1_val,
            "optimal_threshold_balanced_acc": best_bal_acc_th,
            "optimal_threshold_balanced_acc_val": best_bal_acc_val,
            "optimal_threshold_youden": best_youden_th,
            "optimal_threshold_youden_val": best_youden_val
        })
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_config, f, indent=4)
        
    # Update central comparison.csv in the parent experiment directory
    if not is_hyperopt:
        try:
            parent_dir = exp_dir.parent
            comparison_path = parent_dir / "comparison.csv"
            
            # Read existing records if the file exists
            records = []
            fieldnames = [
                "Experiment",
                "Architecture",
                "Loss_Function",
                "Val_Loss",
                "Val_Accuracy",
                "ROC_AUC",
                "PR_AUC",
                "F1",
                "Balanced_Accuracy",
                "Sensitivity",
                "Specificity",
                "Training_Time",
                "Epoch_Time",
                "Parameter_Count",
                "Model_Size",
                "Best_Epoch",
                "Optimizer",
                "Learning_Rate",
                "Weight_Decay",
                "Dropout_Prob",
                "Label_Smoothing",
                "Class_Weights",
                "Augmentation",
                "Augmentation_Profile",
                "Initialization_Type"
            ]
            
            import csv
            existing_names = set()
            if comparison_path.exists():
                try:
                    with open(comparison_path, "r", newline="", encoding="utf-8") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            records.append(row)
                            existing_names.add(row.get("Experiment"))
                except Exception as e:
                    logger.warning(f"Failed to read existing comparison.csv: {e}")
                    
            # Prepare the new record
            new_record = {
                "Experiment": experiment_name,
                "Architecture": architecture,
                "Loss_Function": loss_function,
                "Val_Loss": f"{best_val_loss:.4f}" if best_val_loss is not None else "N/A",
                "Val_Accuracy": f"{best_val_accuracy:.4f}" if best_val_accuracy is not None else "N/A",
                "ROC_AUC": f"{roc_auc:.4f}" if (len(np.unique(y_true)) > 1 and 'roc_auc' in locals()) else "N/A",
                "PR_AUC": f"{avg_precision:.4f}" if (len(np.unique(y_true)) > 1 and avg_precision is not None) else "N/A",
                "F1": f"{report.get('macro avg', {}).get('f1-score', 0.0):.4f}" if 'report' in locals() else "N/A",
                "Balanced_Accuracy": f"{val_bal_acc:.4f}" if 'val_bal_acc' in locals() else "N/A",
                "Sensitivity": f"{sensitivity:.4f}" if 'sensitivity' in locals() else "N/A",
                "Specificity": f"{specificity:.4f}" if 'specificity' in locals() else "N/A",
                "Training_Time": f"{training_time:.2f}" if 'training_time' in locals() else "N/A",
                "Epoch_Time": f"{epoch_time:.2f}" if 'epoch_time' in locals() else "N/A",
                "Parameter_Count": str(total_params) if 'total_params' in locals() else "N/A",
                "Model_Size": f"{model_size_mb:.2f}" if 'model_size_mb' in locals() else "N/A",
                "Best_Epoch": str(best_epoch) if best_epoch is not None else "N/A",
                "Optimizer": optimizer_name,
                "Learning_Rate": str(lr),
                "Weight_Decay": str(weight_decay),
                "Dropout_Prob": str(dropout_prob),
                "Label_Smoothing": str(label_smoothing),
                "Class_Weights": str(use_class_weights),
                "Augmentation": str(augment),
                "Augmentation_Profile": augmentation_profile if augment else "none",
                "Initialization_Type": pretrained_source
            }
            
            # Override if experiment already exists, else append
            if experiment_name in existing_names:
                for idx, rec in enumerate(records):
                    if rec.get("Experiment") == experiment_name:
                        records[idx] = new_record
                        break
            else:
                records.append(new_record)
                
            with open(comparison_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for rec in records:
                    filtered_rec = {k: rec.get(k, "N/A") for k in fieldnames}
                    writer.writerow(filtered_rec)
                    
            logger.info(f"Updated experiment comparison table at: {comparison_path}")
        except Exception as e:
            logger.error(f"Failed to update comparison.csv: {e}")
        
    # 7. ONNX Model Auto-Export
    if not is_hyperopt:
        try:
            onnx_path = exp_dir / "best_model.onnx"
            sample_shape = train_dataset[0].image.shape # e.g. (1, 128, 128, 128)
            dummy_shape = (1,) + tuple(sample_shape)
            dummy_input = torch.randn(dummy_shape).to(device)
            logger.info(f"Auto-exporting best checkpoint model to ONNX: {onnx_path}")
            
            # Export
            export_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.onnx.export(
                export_model,
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
    if copy_outputs_to is not None and not is_hyperopt:
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

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_val_pr_auc": best_val_pr_auc if 'best_val_pr_auc' in locals() else None,
        "best_val_roc_auc": roc_auc if (len(np.unique(y_true)) > 1 and 'roc_auc' in locals()) else None,
        "macro_f1": report.get('macro avg', {}).get('f1-score', 0.0) if 'report' in locals() else 0.0,
    }


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
    parser.add_argument("--dropout-prob", type=float, default=0.0, help="Dropout probability for DenseNet backbone blocks")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing regularization parameter")
    parser.add_argument("--use-class-weights", action="store_true", help="Enable dynamic class weighting for CrossEntropyLoss")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Number of Optuna trials for hyperparameter optimization search (0 to disable)")
    parser.add_argument("--augmentation-profile", default="moderate", choices=["minimal", "moderate", "strong", "research"], help="MONAI augmentation profile to apply when training")
    parser.add_argument("--architecture", default="densenet121", choices=["densenet121", "resnet10", "resnet18"], help="3D CNN architecture backbone model to train")
    parser.add_argument("--benchmark-all", action="store_true", help="Sequentially train densenet121, resnet10, and resnet18 and aggregate comparison metrics")
    parser.add_argument("--loss-function", default="ce", choices=["ce", "weighted_ce", "focal", "ce_ls", "focal_ls"], help="Loss function criterion for training")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma focusing parameter for Focal Loss")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Alpha balancing parameter for Focal Loss")
    parser.add_argument("--benchmark-losses", action="store_true", help="Sequentially train models under ce, weighted_ce, focal, ce_ls, focal_ls criteria and plot ranked statistics")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation during final evaluation")
    parser.add_argument("--tta-runs", type=int, default=5, help="Number of TTA runs to average")
    parser.add_argument("--tta-method", default="mean", choices=["mean", "median", "majority"], help="Probability aggregation method for TTA")
    parser.add_argument("--pretrained", action="store_true", help="Enable pretrained weight initialization")
    parser.add_argument("--pretrained-source", default="none", choices=["none", "medicalnet", "monai", "custom"], help="Source of pretrained weight parameters")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Path to local pretrained model checkpoint file")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone weights and train only classifier for initial epochs")
    parser.add_argument("--freeze-epochs", type=int, default=0, help="Number of epochs to keep backbone frozen")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Differential learning rate for backbone parameters")
    parser.add_argument("--classifier-lr", type=float, default=None, help="Differential learning rate for classifier parameters")
    parser.add_argument("--benchmark-transfer", action="store_true", help="Sequentially train models under random init vs medicalnet vs monai init")

    args = parser.parse_args()

    # Consolidate transfer learning settings to pass to experiments
    extra_run_args = {
        "pretrained": args.pretrained or (args.pretrained_source != "none"),
        "pretrained_source": args.pretrained_source,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "freeze_backbone": args.freeze_backbone,
        "freeze_epochs": args.freeze_epochs,
        "backbone_lr": args.backbone_lr,
        "classifier_lr": args.classifier_lr,
    }

    if args.benchmark_losses:
        from training.losses import aggregate_losses_benchmark
        losses_to_test = ["ce", "weighted_ce", "focal", "ce_ls", "focal_ls"]
        logger.info(f"Starting Multi-Loss Function Benchmark Run for: {losses_to_test}")
        
        for loss in losses_to_test:
            logger.info(f"\n==================================================")
            logger.info(f"TRAINING LOSS FUNCTION: {loss}")
            logger.info(f"==================================================")
            
            # Setup a subfolder per loss to isolate outputs
            loss_dir = Path(args.experiment_dir) / loss
            
            run_training_experiment(
                data_root=args.data_root,
                index_file=args.index_file,
                split_file=args.split_file,
                preprocessed_dir=args.preprocessed_dir,
                config_yaml=args.config_yaml,
                experiment_dir=loss_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                lr=args.lr,
                resume_from="none",
                skip_preprocess=True,
                copy_outputs_to=None,
                strict_cache_validation=args.strict_cache_validation,
                limit=args.limit,
                full_cache_validation=False,
                local_cache_dir=args.local_cache_dir,
                augment=args.augment,
                optimizer_name=args.optimizer,
                weight_decay=args.weight_decay,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                seed=args.seed,
                experiment_name=f"{args.experiment_name}_{loss}",
                decision_threshold=args.decision_threshold,
                early_stopping_patience=args.early_stopping_patience,
                dropout_prob=args.dropout_prob,
                label_smoothing=args.label_smoothing,
                use_class_weights=args.use_class_weights,
                is_hyperopt=False,
                augmentation_profile=args.augmentation_profile,
                architecture=args.architecture,
                loss_function=loss,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                tta=args.tta,
                tta_runs=args.tta_runs,
                tta_method=args.tta_method,
                **extra_run_args,
            )
            
        aggregate_losses_benchmark(Path(args.experiment_dir), losses_to_test)
        logger.info(f"Multi-Loss Benchmark Run completed successfully!")
        
    elif args.benchmark_all:
        from training.benchmark import aggregate_architecture_benchmark
        architectures = ["densenet121", "resnet10", "resnet18"]
        logger.info(f"Starting Multi-Architecture Benchmark Run for: {architectures}")
        
        for arch in architectures:
            logger.info(f"\n==================================================")
            logger.info(f"TRAINING ARCHITECTURE: {arch}")
            logger.info(f"==================================================")
            
            # Setup a subfolder per architecture to isolate outputs
            arch_dir = Path(args.experiment_dir) / arch
            
            run_training_experiment(
                data_root=args.data_root,
                index_file=args.index_file,
                split_file=args.split_file,
                preprocessed_dir=args.preprocessed_dir,
                config_yaml=args.config_yaml,
                experiment_dir=arch_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                lr=args.lr,
                resume_from="none",
                skip_preprocess=True,
                copy_outputs_to=None,
                strict_cache_validation=args.strict_cache_validation,
                limit=args.limit,
                full_cache_validation=False,
                local_cache_dir=args.local_cache_dir,
                augment=args.augment,
                optimizer_name=args.optimizer,
                weight_decay=args.weight_decay,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                seed=args.seed,
                experiment_name=f"{args.experiment_name}_{arch}",
                decision_threshold=args.decision_threshold,
                early_stopping_patience=args.early_stopping_patience,
                dropout_prob=args.dropout_prob,
                label_smoothing=args.label_smoothing,
                use_class_weights=args.use_class_weights,
                is_hyperopt=False,
                augmentation_profile=args.augmentation_profile,
                architecture=arch,
                loss_function=args.loss_function,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                tta=args.tta,
                tta_runs=args.tta_runs,
                tta_method=args.tta_method,
                **extra_run_args,
            )
            
        aggregate_architecture_benchmark(Path(args.experiment_dir), architectures)
        logger.info(f"Multi-Architecture Benchmark Run completed successfully!")
        
    elif args.benchmark_transfer:
        from training.transfer_learning import aggregate_transfer_learning_benchmark
        init_strategies = ["none", "medicalnet", "monai"]
        logger.info(f"Starting Transfer Learning Initialization Benchmark Run for: {init_strategies}")
        
        for init in init_strategies:
            logger.info(f"\n==================================================")
            logger.info(f"TRAINING INITIALIZATION STRATEGY: {init}")
            logger.info(f"==================================================")
            
            init_dir = Path(args.experiment_dir) / init
            
            run_training_experiment(
                data_root=args.data_root,
                index_file=args.index_file,
                split_file=args.split_file,
                preprocessed_dir=args.preprocessed_dir,
                config_yaml=args.config_yaml,
                experiment_dir=init_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                lr=args.lr,
                resume_from="none",
                skip_preprocess=True,
                copy_outputs_to=None,
                strict_cache_validation=args.strict_cache_validation,
                limit=args.limit,
                full_cache_validation=False,
                local_cache_dir=args.local_cache_dir,
                augment=args.augment,
                optimizer_name=args.optimizer,
                weight_decay=args.weight_decay,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                seed=args.seed,
                experiment_name=f"{args.experiment_name}_{init}",
                decision_threshold=args.decision_threshold,
                early_stopping_patience=args.early_stopping_patience,
                dropout_prob=args.dropout_prob,
                label_smoothing=args.label_smoothing,
                use_class_weights=args.use_class_weights,
                is_hyperopt=False,
                augmentation_profile=args.augmentation_profile,
                architecture=args.architecture,
                loss_function=args.loss_function,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                tta=args.tta,
                tta_runs=args.tta_runs,
                tta_method=args.tta_method,
                pretrained=(init != "none"),
                pretrained_source=init,
                pretrained_checkpoint=args.pretrained_checkpoint,
                freeze_backbone=args.freeze_backbone,
                freeze_epochs=args.freeze_epochs,
                backbone_lr=args.backbone_lr,
                classifier_lr=args.classifier_lr,
            )
            
        aggregate_transfer_learning_benchmark(Path(args.experiment_dir), init_strategies)
        logger.info(f"Transfer Learning Benchmark Run completed successfully!")

    elif args.optuna_trials > 0:
        from training.hyperopt import optimize_hyperparameters
        import shutil
        
        def objective(trial):
            # Suggest hyperparams
            trial_lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            trial_weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            trial_batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24])
            trial_dropout = trial.suggest_float("dropout_prob", 0.0, 0.5)
            trial_patience = trial.suggest_int("scheduler_patience", 3, 10)
            trial_factor = trial.suggest_float("scheduler_factor", 0.1, 0.7)
            
            # Setup a unique trial directory to isolate runs
            trial_dir = Path(args.experiment_dir) / f"trial_{trial.number}"
            
            try:
                metrics = run_training_experiment(
                    data_root=args.data_root,
                    index_file=args.index_file,
                    split_file=args.split_file,
                    preprocessed_dir=args.preprocessed_dir,
                    config_yaml=args.config_yaml,
                    experiment_dir=trial_dir,
                    epochs=args.epochs,
                    batch_size=trial_batch_size,
                    device=args.device,
                    lr=trial_lr,
                    resume_from="none",
                    skip_preprocess=args.skip_preprocess,
                    copy_outputs_to=None,
                    strict_cache_validation=args.strict_cache_validation,
                    limit=args.limit,
                    full_cache_validation=args.full_cache_validation,
                    local_cache_dir=args.local_cache_dir,
                    augment=args.augment,
                    optimizer_name=args.optimizer,
                    weight_decay=trial_weight_decay,
                    scheduler_patience=trial_patience,
                    scheduler_factor=trial_factor,
                    seed=args.seed,
                    experiment_name=f"{args.experiment_name}_trial_{trial.number}",
                    decision_threshold=args.decision_threshold,
                    early_stopping_patience=args.early_stopping_patience,
                    dropout_prob=trial_dropout,
                    label_smoothing=args.label_smoothing,
                    use_class_weights=args.use_class_weights,
                    is_hyperopt=True,
                    augmentation_profile=args.augmentation_profile,
                    architecture=args.architecture,
                    loss_function=args.loss_function,
                    focal_gamma=args.focal_gamma,
                    focal_alpha=args.focal_alpha,
                    tta=False,
                    tta_runs=1,
                    tta_method="mean",
                    **extra_run_args,
                )
                
                # Fetch ROC-AUC as optimization target
                roc_auc = metrics.get("best_val_roc_auc")
                if roc_auc is None:
                    roc_auc = metrics.get("best_val_accuracy", 0.0)
                    
                return roc_auc
                
            finally:
                if trial_dir.exists():
                    try:
                        shutil.rmtree(trial_dir)
                    except Exception as e:
                        logger.warning(f"Could not delete trial directory {trial_dir}: {e}")
                        
        best_results = optimize_hyperparameters(
            objective_fn=objective,
            n_trials=args.optuna_trials,
            seed=args.seed,
            output_dir=Path(args.experiment_dir)
        )
        
        best_params = best_results["best_params"]
        logger.info(f"Optimization completed. Best hyperparameters: {best_params}")
        logger.info("Starting final training run using the best hyperparameters...")
        
        run_training_experiment(
            data_root=args.data_root,
            index_file=args.index_file,
            split_file=args.split_file,
            preprocessed_dir=args.preprocessed_dir,
            config_yaml=args.config_yaml,
            experiment_dir=args.experiment_dir,
            epochs=args.epochs,
            batch_size=best_params["batch_size"],
            device=args.device,
            lr=best_params["lr"],
            resume_from=args.resume_from,
            skip_preprocess=args.skip_preprocess,
            copy_outputs_to=args.copy_outputs_to,
            strict_cache_validation=args.strict_cache_validation,
            limit=args.limit,
            full_cache_validation=args.full_cache_validation,
            local_cache_dir=args.local_cache_dir,
            augment=args.augment,
            optimizer_name=args.optimizer,
            weight_decay=best_params["weight_decay"],
            scheduler_patience=best_params["scheduler_patience"],
            scheduler_factor=best_params["scheduler_factor"],
            seed=args.seed,
            experiment_name=args.experiment_name,
            decision_threshold=args.decision_threshold,
            early_stopping_patience=args.early_stopping_patience,
            dropout_prob=best_params["dropout_prob"],
            label_smoothing=args.label_smoothing,
            use_class_weights=args.use_class_weights,
            is_hyperopt=False,
            augmentation_profile=args.augmentation_profile,
            architecture=args.architecture,
            loss_function=args.loss_function,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            tta=args.tta,
            tta_runs=args.tta_runs,
            tta_method=args.tta_method,
            **extra_run_args,
        )
    else:
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
            dropout_prob=args.dropout_prob,
            label_smoothing=args.label_smoothing,
            use_class_weights=args.use_class_weights,
            is_hyperopt=False,
            augmentation_profile=args.augmentation_profile,
            architecture=args.architecture,
            loss_function=args.loss_function,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            tta=args.tta,
            tta_runs=args.tta_runs,
            tta_method=args.tta_method,
            **extra_run_args,
        )
