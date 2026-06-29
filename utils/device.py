"""Backend-agnostic device resolver and hardware capabilities mapping for NeuroFramework."""

import importlib.metadata
import sys
import warnings
from dataclasses import dataclass
import torch


@dataclass
class DeviceCapabilities:
    """Hardware optimization flags mapping for different hardware accelerators."""
    amp: bool
    pin_memory: bool
    non_blocking: bool
    benchmark: bool


@dataclass
class ResolvedBackend:
    """Device description packaging a PyTorch device, name, version, and capabilities."""
    device: torch.device
    name: str
    version: str
    capabilities: DeviceCapabilities


def check_directml_compatibility() -> str:
    """Verifies torch-directml availability and PyTorch version compatibility constraints.

    Returns:
        The version string of the installed torch-directml package.
    """
    try:
        import torch_directml
    except ImportError as e:
        torch_version = torch.__version__
        raise ImportError(
            f"Failed to import torch_directml: {e}.\n"
            f"Please ensure you have installed torch-directml. Local AMD GPU training requires "
            f"a compatible PyTorch version (e.g., PyTorch 2.1.x or 2.2.x). "
            f"Currently installed PyTorch version: {torch_version}.\n"
            f"You can install it by running: pip install -r requirements-directml.txt"
        ) from e

    try:
        directml_version = importlib.metadata.version("torch-directml")
    except Exception:
        directml_version = "unknown"

    # DirectML version warnings for newer PyTorch versions
    parts = torch.__version__.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        major = 2
        minor = 1

    if major > 2 or (major == 2 and minor > 2):
        warnings.warn(
            f"Installed PyTorch version ({torch.__version__}) is newer than the officially supported "
            f"compatibility range for torch-directml (PyTorch 2.0.x - 2.2.x). "
            f"You may experience runtime stability issues.",
            RuntimeWarning
        )

    return directml_version


def resolve_backend(device_name: str | torch.device) -> ResolvedBackend:
    """Resolves a target device identifier into a structured ResolvedBackend object.

    Args:
        device_name: The device identifier (e.g., 'auto', 'cpu', 'cuda', 'directml').
    """
    if isinstance(device_name, torch.device):
        dev = device_name
        name = dev.type
        if name == "cuda":
            return ResolvedBackend(
                device=dev,
                name="cuda",
                version=torch.version.cuda or "unknown",
                capabilities=DeviceCapabilities(amp=True, pin_memory=True, non_blocking=True, benchmark=True)
            )
        elif name == "privateuseone":
            try:
                dml_version = importlib.metadata.version("torch-directml")
            except Exception:
                dml_version = "unknown"
            return ResolvedBackend(
                device=dev,
                name="directml",
                version=dml_version,
                capabilities=DeviceCapabilities(amp=False, pin_memory=False, non_blocking=False, benchmark=False)
            )
        else:
            return ResolvedBackend(
                device=dev,
                name=name,
                version="N/A",
                capabilities=DeviceCapabilities(amp=False, pin_memory=False, non_blocking=False, benchmark=False)
            )

    dev_str = str(device_name).lower().strip()

    if dev_str == "auto":
        # 1. CUDA
        if torch.cuda.is_available():
            return resolve_backend("cuda")
        # 2. DirectML
        try:
            check_directml_compatibility()
            return resolve_backend("directml")
        except Exception:
            pass
        # 3. MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return resolve_backend("mps")
        # 4. CPU
        return resolve_backend("cpu")

    elif dev_str == "cuda":
        return ResolvedBackend(
            device=torch.device("cuda"),
            name="cuda",
            version=torch.version.cuda or "unknown",
            capabilities=DeviceCapabilities(amp=True, pin_memory=True, non_blocking=True, benchmark=True)
        )

    elif dev_str == "directml":
        dml_ver = check_directml_compatibility()
        import torch_directml
        return ResolvedBackend(
            device=torch_directml.device(),
            name="directml",
            version=dml_ver,
            capabilities=DeviceCapabilities(amp=False, pin_memory=False, non_blocking=False, benchmark=False)
        )

    elif dev_str == "mps":
        return ResolvedBackend(
            device=torch.device("mps"),
            name="mps",
            version="N/A",
            capabilities=DeviceCapabilities(amp=False, pin_memory=False, non_blocking=False, benchmark=False)
        )

    elif dev_str == "xpu":
        return ResolvedBackend(
            device=torch.device("xpu"),
            name="xpu",
            version="N/A",
            capabilities=DeviceCapabilities(amp=False, pin_memory=False, non_blocking=False, benchmark=False)
        )

    elif dev_str == "cpu":
        return ResolvedBackend(
            device=torch.device("cpu"),
            name="cpu",
            version="N/A",
            capabilities=DeviceCapabilities(amp=False, pin_memory=False, non_blocking=False, benchmark=False)
        )

    else:
        # Support index specifications (e.g. 'cuda:0')
        if dev_str.startswith("cuda"):
            return ResolvedBackend(
                device=torch.device(dev_str),
                name="cuda",
                version=torch.version.cuda or "unknown",
                capabilities=DeviceCapabilities(amp=True, pin_memory=True, non_blocking=True, benchmark=True)
            )
        try:
            dev = torch.device(dev_str)
            name = "directml" if dev.type == "privateuseone" else dev.type
            version = "N/A"
            if name == "directml":
                try:
                    version = importlib.metadata.version("torch-directml")
                except Exception:
                    version = "unknown"
            return ResolvedBackend(
                device=dev,
                name=name,
                version=version,
                capabilities=DeviceCapabilities(amp=False, pin_memory=False, non_blocking=False, benchmark=False)
            )
        except Exception as e:
            raise ValueError(f"Unknown device identifier: '{device_name}'. Error: {e}")


def load_state_dict_flexible(model: torch.nn.Module, state_dict: dict) -> None:
    """Loads a state dict into a PyTorch model, dynamically handling 'module.' prefixes

    from DataParallel wrappers.
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    # Check if model has a 'module' attribute (is wrapped in DataParallel)
    model_has_module = hasattr(model, "module") or isinstance(model, torch.nn.DataParallel)
    
    for k, v in state_dict.items():
        is_prefixed = k.startswith("module.")
        if model_has_module and not is_prefixed:
            new_state_dict[f"module.{k}"] = v
        elif not model_has_module and is_prefixed:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    print("=" * 40)
    print("NeuroFramework Device Report")
    print("=" * 40)
    print(f"PyTorch: {torch.__version__}")
    print()
    print("CUDA:")
    print(f"  Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Version: {torch.version.cuda}")
        try:
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    dml_installed = False
    dml_available = False
    dml_version = "N/A"
    try:
        import torch_directml
        dml_installed = True
        dml_available = True
        try:
            dml_version = importlib.metadata.version("torch-directml")
        except Exception:
            dml_version = "unknown"
    except ImportError:
        pass

    print()
    print("DirectML:")
    print(f"  Installed: {dml_installed}")
    print(f"  Available: {dml_available}")
    if dml_installed:
        print(f"  Version: {dml_version}")

    mps_available = False
    if hasattr(torch.backends, "mps"):
        mps_available = torch.backends.mps.is_available()
    print()
    print("MPS:")
    print(f"  Available: {mps_available}")

    backend = resolve_backend("auto")
    print()
    print("Selected Backend:")
    print(f"  Name: {backend.name}")
    print(f"  Device: {backend.device}")
    print(f"  Version: {backend.version}")

    print()
    print("Capabilities:")
    print(f"  AMP: {'Yes' if backend.capabilities.amp else 'No'}")
    print(f"  Pin Memory: {'Yes' if backend.capabilities.pin_memory else 'No'}")
    print(f"  Non Blocking: {'Yes' if backend.capabilities.non_blocking else 'No'}")
    print(f"  Benchmark: {'Yes' if backend.capabilities.benchmark else 'No'}")
    print("=" * 40)
