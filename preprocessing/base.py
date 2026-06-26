"""Abstract base transform interface definitions and global transform registry factory."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type
from schemas.mri import MRIData
from schemas.processing import ExecutionContext


class BaseTransform(ABC):
    """Abstract Base Class for all preprocessing steps. Enforces a unified config signature."""

    def __init__(self, **kwargs: Any):
        self.params = kwargs

    @abstractmethod
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        """Executes preprocessing logic, returning modified copy of MRIData."""
        pass


class BaseTransformStrategy(ABC):
    """Abstract Base Class for toolkit-specific algorithms (e.g. SimpleITK, MONAI)."""

    @abstractmethod
    def execute(self, mri_data: MRIData, context: ExecutionContext, params: Dict[str, Any]) -> MRIData:
        """Executes strategy algorithm block on the MRIData object."""
        pass


class TransformRegistry:
    """Registry managing preprocessing transform mappings dynamically."""

    _registry: Dict[str, Type[BaseTransform]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseTransform]], Type[BaseTransform]]:
        """Decorator to register a new Preprocessing Transform."""
        def decorator(subclass: Type[BaseTransform]) -> Type[BaseTransform]:
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseTransform:
        """Instantiates registered transform using config parameters."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(f"Transform '{name}' is not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name_lower](**kwargs)


def mri_data_to_sitk(mri_data: MRIData) -> "sitk.Image":
    """Converts active MRIData tensor/affine to a SimpleITK image."""
    import SimpleITK as sitk
    import numpy as np

    tensor = mri_data.image if mri_data.image is not None else mri_data.raw.tensor
    
    if len(tensor.shape) == 4:
        if tensor.shape[0] == 1:
            tensor_3d = tensor[0]
        else:
            raise ValueError("SimpleITK transforms only support 3D images (1 channel).")
    else:
        tensor_3d = tensor
        
    sitk_img = sitk.GetImageFromArray(np.transpose(tensor_3d, (2, 1, 0)))
    
    affine = mri_data.affine if mri_data.affine is not None else mri_data.raw.affine
    origin = [float(x) for x in affine[:3, 3]]
    spacing = [float(np.linalg.norm(affine[:3, i])) for i in range(3)]
    
    dir_matrix = np.zeros((3, 3))
    for i in range(3):
        if spacing[i] > 0:
            dir_matrix[:, i] = affine[:3, i] / spacing[i]
        else:
            dir_matrix[:, i] = affine[:3, i]
            
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetDirection(dir_matrix.flatten().tolist())
    
    return sitk_img


def sitk_to_mri_data(sitk_img: "sitk.Image", original_mri_data: MRIData) -> MRIData:
    """Converts a SimpleITK Image back to standard MRIData, updating image and affine."""
    import SimpleITK as sitk
    import numpy as np

    arr = sitk.GetArrayFromImage(sitk_img)
    tensor_3d = np.transpose(arr, (2, 1, 0))
    
    orig_tensor = original_mri_data.image if original_mri_data.image is not None else original_mri_data.raw.tensor
    if len(orig_tensor.shape) == 4:
        tensor = np.expand_dims(tensor_3d, axis=0)
    else:
        tensor = tensor_3d
        
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    
    dir_matrix = np.array(direction).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = dir_matrix * np.array(spacing)
    affine[:3, 3] = origin
    
    mri_copy = original_mri_data.model_copy(deep=True)
    mri_copy.image = tensor
    mri_copy.affine = affine
    mri_copy.metadata.image.voxel_dims = list(spacing)
    mri_copy.metadata.image.dimensions = list(tensor.shape)
    
    return mri_copy
