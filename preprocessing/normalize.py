"""MRI intensity normalization implementing Z-Score and Min-Max scaling."""

import numpy as np
from typing import Any
from preprocessing.base import BaseTransform, TransformRegistry
from preprocessing.decorators import trace_transform
from schemas.mri import MRIData, ScanStatistics
from schemas.processing import ExecutionContext


@TransformRegistry.register("normalize")
class IntensityNormalizer(BaseTransform):
    """Normalizes MRI intensity values using Z-score or Min-Max scaling strategies."""

    def __init__(self, mode: str = "z_score", **kwargs: Any):
        super().__init__(mode=mode, **kwargs)

    @trace_transform
    def process(self, mri_data: MRIData, context: ExecutionContext) -> MRIData:
        tensor = mri_data.image
        mode = self.params.get("mode", "z_score")
        
        if mode == "z_score":
            mean = np.mean(tensor)
            std = np.std(tensor)
            new_tensor = (tensor - mean) / (std + 1e-8)
        elif mode == "min_max":
            t_min = np.min(tensor)
            t_max = np.max(tensor)
            new_tensor = (tensor - t_min) / (t_max - t_min + 1e-8)
        else:
            raise ValueError(f"IntensityNormalizer: Unsupported normalization mode '{mode}'. Use 'z_score' or 'min_max'.")
            
        mri_copy = mri_data.model_copy(deep=True)
        mri_copy.image = new_tensor
        
        # Recalculate statistics
        mri_copy.statistics = ScanStatistics(
            min=float(np.min(new_tensor)),
            max=float(np.max(new_tensor)),
            mean=float(np.mean(new_tensor)),
            std=float(np.std(new_tensor)),
            shape=list(new_tensor.shape),
            dtype=str(new_tensor.dtype)
        )
        
        return mri_copy
