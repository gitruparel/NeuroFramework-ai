"""Unit and integration tests for ModelFactory and 3D architectures."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from models.factory import ModelFactory
from models.densenet import DenseNet3D
from models.resnet import ResNet3D
from training.benchmark import generate_model_summary


def test_factory_creates_models():
    """Verify that all supported models build through the factory."""
    models = ["densenet121", "resnet10", "resnet18"]
    for arch in models:
        model = ModelFactory.create_model(
            model_name=arch,
            in_channels=1,
            out_channels=2,
            dropout_prob=0.1
        )
        assert model is not None
        assert isinstance(model, nn.Module)
        
        # Verify specific classes
        if arch == "densenet121":
            assert isinstance(model, DenseNet3D)
        else:
            assert isinstance(model, ResNet3D)


def test_forward_pass_identical_output_shape():
    """Verify that forward pass executes successfully and returns expected dimensions."""
    dummy_input = torch.randn(2, 1, 32, 32, 32) # batch size 2, channels 1, shape 32^3
    
    models = ["densenet121", "resnet10", "resnet18"]
    for arch in models:
        model = ModelFactory.create_model(
            model_name=arch,
            in_channels=1,
            out_channels=2,
            dropout_prob=0.0
        )
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            
        assert output.shape == (2, 2)
        assert output.dtype == torch.float32


def test_parameter_counts():
    """Verify parameter calculations and sizes are non-zero."""
    model = ModelFactory.create_model("resnet10", in_channels=1, out_channels=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


def test_model_summary_generation(tmp_path):
    """Verify that model_summary.txt is created successfully and contains parameter estimates."""
    model = ModelFactory.create_model("resnet10", in_channels=1, out_channels=2)
    summary_path = tmp_path / "model_summary.txt"
    
    # Generate summary
    input_shape = (1, 1, 32, 32, 32)
    generate_model_summary(model, input_shape, summary_path)
    
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    
    # Verify expected keywords
    assert "Model Summary: ResNet3D" in content
    assert "Total Parameters" in content
    assert "Trainable Parameters" in content
    assert "Estimated Parameter Size" in content
    assert "Estimated Activation Memory Footprint" in content
