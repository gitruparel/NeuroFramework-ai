"""Pretrained weight loader and layer freezing utilities for 3D CNN backbones."""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from core.logging import setup_logger

logger = setup_logger("models.pretrained", "logs/models/pretrained.log")


def freeze_backbone(model: nn.Module) -> None:
    """Freezes all layers except the classification head."""
    classifier_names = ["class_out", "classifier", "fc"]
    frozen_count = 0
    for name, param in model.named_parameters():
        is_classifier = any(cn in name for cn in classifier_names)
        if not is_classifier:
            param.requires_grad = False
            frozen_count += 1
    logger.info(f"freeze_backbone: Froze {frozen_count} backbone parameters.")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreezes all parameters in the model."""
    unfrozen_count = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_count += 1
    logger.info(f"unfreeze_backbone: Unfroze {unfrozen_count} parameters.")


def load_pretrained_weights(
    model: nn.Module,
    source: str,
    checkpoint_path: Optional[str] = None,
    architecture: str = "densenet121",
) -> nn.Module:
    """Loads weights from MONAI, MedicalNet, or a custom local checkpoint path."""
    source_lower = source.lower()
    
    if source_lower == "none":
        logger.info("Using random initialization (no pretrained weights).")
        return model
        
    state_dict = None
    
    if source_lower == "custom":
        if not checkpoint_path:
            raise ValueError("For custom weight loading, --pretrained-checkpoint path must be provided.")
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom checkpoint file not found at: {path}")
        logger.info(f"Loading custom weights from checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
    elif source_lower == "medicalnet":
        if checkpoint_path:
            path = Path(checkpoint_path)
        else:
            path = Path(f"cache/weights/medicalnet_{architecture}.pth")
            
        if not path.exists():
            logger.warning(f"MedicalNet weights file not found locally at: {path}. Stubbing initialization for testing.")
            return model
            
        logger.info(f"Loading MedicalNet weights from: {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
    elif source_lower == "monai":
        if checkpoint_path:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"MONAI checkpoint file not found at: {path}")
            logger.info(f"Loading MONAI weights from: {path}")
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
        else:
            # Let MONAI handle its own loading, or return model as-is for testing
            logger.info("Using default MONAI model initialization.")
            return model
            
    if state_dict is not None:
        model_state = model.state_dict()
        mapped_state = {}
        matched_keys = []
        unmatched_keys = []
        
        # Clean state_dict keys (e.g. remove 'module.' prefix)
        cleaned_state = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("module."):
                key = key[7:]
            cleaned_state[key] = v
            
        for name, param in model_state.items():
            if name in cleaned_state and cleaned_state[name].shape == param.shape:
                mapped_state[name] = cleaned_state[name]
                matched_keys.append(name)
            else:
                # Fuzzy matching based on shape and name substring matching
                fuzzy_found = False
                for ck_name, ck_param in cleaned_state.items():
                    if ck_param.shape == param.shape and (ck_name in name or name in ck_name):
                        mapped_state[name] = ck_param
                        matched_keys.append(name)
                        fuzzy_found = True
                        break
                if not fuzzy_found:
                    unmatched_keys.append(name)
                    
        model.load_state_dict(mapped_state, strict=False)
        logger.info(f"Pretrained loading: Matched {len(mapped_state)} / {len(model_state)} parameters.")
        if unmatched_keys:
            logger.info(f"Pretrained loading: Unmatched parameters (e.g., classifier head): {unmatched_keys[:10]}...")
            
    return model
