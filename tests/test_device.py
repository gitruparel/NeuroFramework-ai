"""Unit tests validating the backend-agnostic resolve_backend resolver and DeviceCapabilities."""

import pytest
import sys
import torch
from unittest.mock import MagicMock, patch

from utils.device import resolve_backend, check_directml_compatibility, DeviceCapabilities


def test_resolve_cpu_and_cuda():
    """Verify that cpu and cuda devices resolve correctly with expected capabilities."""
    cpu_backend = resolve_backend("cpu")
    assert cpu_backend.device.type == "cpu"
    assert cpu_backend.name == "cpu"
    assert cpu_backend.capabilities.amp is False
    assert cpu_backend.capabilities.pin_memory is False

    cuda_backend = resolve_backend("cuda")
    assert cuda_backend.device.type == "cuda"
    assert cuda_backend.name == "cuda"
    assert cuda_backend.capabilities.amp is True
    assert cuda_backend.capabilities.pin_memory is True


def test_resolve_auto_priority_cuda():
    """Verify auto resolution defaults to cuda if CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True):
        backend = resolve_backend("auto")
        assert backend.name == "cuda"
        assert backend.capabilities.amp is True


def test_resolve_auto_priority_directml():
    """Verify auto resolution defaults to directml if CUDA is unavailable but directml is present."""
    # Mock directml compatibility check and import
    mock_torch_directml = MagicMock()
    mock_torch_directml.device.return_value = torch.device("cpu")  # mock device
    
    with patch("torch.cuda.is_available", return_value=False), \
         patch("utils.device.check_directml_compatibility", return_value="0.1.0"), \
         patch.dict("sys.modules", {"torch_directml": mock_torch_directml}):
         
        backend = resolve_backend("auto")
        assert backend.name == "directml"
        assert backend.version == "0.1.0"
        assert backend.capabilities.amp is False


def test_resolve_auto_priority_cpu_fallback():
    """Verify auto resolution falls back to cpu if no accelerators are present."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("utils.device.check_directml_compatibility", side_effect=ImportError("mock")), \
         patch("torch.backends.mps.is_available", return_value=False):
         
        backend = resolve_backend("auto")
        assert backend.name == "cpu"
        assert backend.capabilities.amp is False


def test_directml_import_failure():
    """Verify that resolving directml explicitly raises ImportError with instructions if not installed."""
    if "torch_directml" in sys.modules:
        del sys.modules["torch_directml"]

    orig_import = __import__
    def import_mock(name, *args, **kwargs):
        if name == "torch_directml":
            raise ImportError("Module not found")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_mock):
        with pytest.raises(ImportError, match="Failed to import torch_directml"):
            resolve_backend("directml")


def test_pytorch_version_warning_directml():
    """Verify that compatibility check warns on newer unsupported PyTorch versions."""
    mock_torch_directml = MagicMock()
    
    with patch.dict("sys.modules", {"torch_directml": mock_torch_directml}), \
         patch("importlib.metadata.version", return_value="0.2.0"), \
         patch("torch.__version__", "2.12.1+cpu"), \
         pytest.warns(RuntimeWarning, match="newer than the officially supported"):
         
        version = check_directml_compatibility()
        assert version == "0.2.0"
