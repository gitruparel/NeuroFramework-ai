"""Unit tests for the research-grade MRI augmentation pipeline."""

import os
from pathlib import Path
import pytest
import numpy as np
import torch

from training.experiment import set_seed
from training.augmentations import (
    get_mri_augmentations,
    get_profile_metadata,
    generate_augmentation_preview
)


def test_augmentation_profile_builds():
    """Verify that all profiles build successfully and return Compose pipelines."""
    for profile in ["minimal", "moderate", "strong", "research"]:
        pipeline = get_mri_augmentations(profile, seed=42)
        assert pipeline is not None
        assert len(pipeline.transforms) > 0
        
        # Verify metadata
        meta = get_profile_metadata(profile)
        assert meta["augmentation_profile"] == profile
        assert len(meta["enabled_transforms"]) == len(pipeline.transforms)


def test_augmentation_output_properties():
    """Verify that output shapes and dtypes are preserved exactly during augmentations."""
    # Build moderate pipeline
    pipeline = get_mri_augmentations("moderate", seed=42)
    
    # Generate mock 3D tensor of shape (1, 16, 16, 16)
    mock_input = torch.randn(1, 16, 16, 16, dtype=torch.float32)
    
    # Run transform
    mock_output = pipeline(mock_input)
    
    # Assertions
    assert mock_output.shape == mock_input.shape
    assert mock_output.dtype == mock_input.dtype


def test_augmentation_determinism():
    """Verify that setting the random seed ensures 100% deterministic outputs."""
    mock_input = torch.randn(1, 16, 16, 16)
    
    # Run 1
    set_seed(42)
    pipeline_1 = get_mri_augmentations("moderate", seed=42)
    output_1 = pipeline_1(mock_input)
    
    # Run 2
    set_seed(42)
    pipeline_2 = get_mri_augmentations("moderate", seed=42)
    output_2 = pipeline_2(mock_input)
    
    # Assert identity
    assert torch.allclose(output_1, output_2, atol=1e-5)


def test_augmentation_preview_generation(tmp_path):
    """Verify that the augmentation visualizer successfully creates previews."""
    mock_input = torch.randn(1, 16, 16, 16)
    pipeline = get_mri_augmentations("moderate", seed=42)
    
    preview_path = tmp_path / "augmentation_preview.png"
    
    # Generate preview
    generate_augmentation_preview(mock_input, pipeline, preview_path)
    
    # Assert file exists and is non-empty
    assert preview_path.exists()
    assert preview_path.stat().st_size > 0
